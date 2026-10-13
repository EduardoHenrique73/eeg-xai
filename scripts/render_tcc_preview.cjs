const fs = require('fs');
const path = require('path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const root = path.resolve(__dirname, '..');
const source = path.join(root, 'docs/tcc/TCC II - Eduardo Marques - Versao Final Revisada.docx');
const output = path.join(root, 'docs/tcc/render_revisao/TCC II - preview sem Word.pdf');
const bundle = path.join(root, '.tcc-preview/node_modules/docx-preview/dist/docx-preview.min.js');
const docxPreviewPackage = fs.realpathSync(path.join(root, '.tcc-preview/node_modules/docx-preview'));
const jszipPackage = require.resolve('jszip/package.json', {
  paths: [docxPreviewPackage],
});
const jszip = path.join(path.dirname(jszipPackage), 'dist/jszip.min.js');

async function main() {
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROME_PATH || undefined,
    args: ['--no-sandbox'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    page.on('pageerror', error => process.stderr.write(`Page error: ${error.message}\n`));
    await page.setContent('<!doctype html><html><head><meta charset="utf-8"></head><body><div id="pages"></div></body></html>');
    await page.addScriptTag({ path: jszip });
    await page.addScriptTag({ path: bundle });
    const data = fs.readFileSync(source).toString('base64');
    const result = await page.evaluate(async encoded => {
      const binary = atob(encoded);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      await docx.renderAsync(bytes, document.getElementById('pages'), null, {
        breakPages: true,
        experimental: true,
        useBase64URL: true,
        ignoreLastRenderedPageBreak: true,
      });
      await document.fonts.ready;
      for (const section of document.querySelectorAll('section.docx')) {
        if (!section.innerText.trim() && !section.querySelector('img, table')) section.remove();
      }
      return {
        sections: document.querySelectorAll('section.docx').length,
        images: document.querySelectorAll('section.docx img').length,
        textLength: document.body.innerText.length,
      };
    }, data);
    await page.addStyleTag({ content: `
      @page { size: A4; margin: 0; }
      html, body { margin: 0; padding: 0; background: white; }
      .docx-wrapper { padding: 0 !important; background: white !important; }
      section.docx { margin: 0 !important; box-shadow: none !important; }
      section.docx:not(:first-child) { break-before: page; }
      section.docx a, section.docx a * { color: black !important; text-decoration: none !important; }
    ` });
    await page.pdf({ path: output, format: 'A4', printBackground: true, preferCSSPageSize: true });
    console.log(JSON.stringify({ output, ...result }));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
