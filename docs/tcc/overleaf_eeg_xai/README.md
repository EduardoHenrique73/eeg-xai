# TCC II no Overleaf

Projeto LaTeX migrado do documento institucional
`TCC_II_Eduardo_Estilo_Karollyne_IFG_Final.docx`, sem depender do Microsoft Word.

## Como abrir

1. No Overleaf, selecione **New Project > Upload Project**.
2. Envie o arquivo `TCC_II_Eduardo_IFG_Overleaf.zip`.
3. Confirme `main.tex` como documento principal.
4. Use o compilador **pdfLaTeX**. XeLaTeX também é compatível.

## Estrutura

- `main.tex`: configuração ABNT, metadados e ordem do documento.
- `pretextuais.tex`: folha de aprovação, agradecimentos, resumo, abstract, listas, siglas e sumário.
- `conteudo.tex`: capítulos, referências e apêndice.
- `images/`: 19 imagens incorporadas no documento original.

## Formatação principal

- Classe `abntex2`, papel A4, corpo 12 pt e tipografia Times.
- Margens superior e esquerda de 3 cm; inferior e direita de 2 cm.
- Recuo de primeira linha de 1,25 cm e espaçamento 1,5.
- Sumário, lista de figuras e numeração gerados pelo LaTeX.

O DOCX institucional original permanece preservado em `docs/tcc` e é a autoridade
para conferência de conteúdo e aparência.
