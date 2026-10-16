# Auditoria de formatação e estilo do projeto Overleaf

Documento de referência: `TCC_II_Eduardo_Estilo_Karollyne_IFG_Final.docx`.

## Resultado geral

O projeto compila, contém os textos, as 19 imagens e as 13 tabelas, mas ainda não reproduz
com fidelidade suficiente o padrão institucional do DOCX. Há falhas estruturais visíveis que
precisam ser corrigidas antes de considerar o arquivo pronto para entrega.

## Status após a correção

Os itens estruturais desta auditoria foram corrigidos em 16 de outubro de 2026:

- as três listas manuais duplicadas foram removidas do corpo;
- foi criada uma Lista de Quadros alimentada pelos dez quadros;
- as fontes duplicadas das 19 figuras foram removidas;
- referências e apêndice passaram para a parte pós-textual;
- o apêndice usa o ambiente `apendicesenv`, com numeração A.1 e A.2;
- legendas e fontes usam corpo 10, centralização e espaçamento simples;
- as dimensões originais das 19 figuras foram restauradas;
- os quadros largos receberam larguras e espaçamentos específicos;
- as principais mesclagens do orçamento e cronograma foram restauradas;
- marcadores consecutivos foram agrupados em sete listas;
- títulos secundários deixaram de usar negrito;
- referências passaram a usar espaçamento simples;
- figuras passaram a aceitar posicionamento `htbp`;
- a fonte foi atualizada para NewTX, uma alternativa Times adequada ao LaTeX.

O projeto recompilou sem erros. Permanece necessária a inspeção visual página a página no PDF,
principalmente para ajustar quebras de página, órfãs/viúvas e o posicionamento final de figuras.

## Prioridade crítica

1. **Listas e sumário duplicados no corpo.** As três primeiras `longtable` de `conteudo.tex`
   reproduzem manualmente a lista de figuras, a lista de quadros e o sumário. Em seguida,
   `pretextuais.tex` já gera essas listas automaticamente. O PDF fica com conteúdo duplicado
   antes da Introdução.
2. **Lista de quadros incorreta e vazia.** O projeto usa `\listoftables`, cujo nome padrão é
   “Lista de Tabelas”, enquanto o documento do IFG usa “LISTA DE QUADROS”. Os dez quadros não
   possuem comandos de legenda associados ao contador, portanto não entram na lista automática.
3. **Fontes das figuras duplicadas.** Cada figura recebe `\fonte{...}` dentro do ambiente e
   conserva também o parágrafo `Fonte: ...` depois dele. Há 19 duplicações.
4. **Referências e apêndice ainda estão na parte textual.** `\postextual` aparece somente depois
   de `\input{conteudo}`. Como referências e apêndice estão dentro desse arquivo, recebem o
   regime tipográfico da parte textual. O apêndice também foi escrito como `\chapter*`, sem o
   ambiente próprio de apêndices; as subseções podem aparecer como continuação do capítulo 7,
   em vez de A.1 e A.2.

## Prioridade alta

5. **Quadros não preservam a aparência do DOCX.** As legendas foram convertidas em texto comum,
   alinhado à esquerda e em negrito. No DOCX, elas são centralizadas, regulares, em Times New
   Roman 10 pt e espaçamento simples. As fontes dos quadros também deveriam ser centralizadas,
   em 10 pt e espaçamento simples.
6. **Legendas de figuras divergentes.** O LaTeX aplica `labelfont=bf`, enquanto o DOCX usa legenda
   regular em 10 pt. A distância entre legenda, figura e fonte também não reproduz os 3 pt e
   7 pt identificados no documento institucional.
7. **Todas as imagens foram redimensionadas para 95% da largura.** O DOCX usa 13,6 cm para a
   maioria das imagens e 8,47 cm para a figura vertical. O LaTeX amplia todas para cerca de
   15,2 cm, alterando hierarquia visual, paginação e nitidez.
8. **Tabelas largas ultrapassam a área útil.** O cronograma possui 13 colunas; o orçamento possui
   seis. A soma das larguras e dos espaçamentos internos ultrapassa `\textwidth`. Esses quadros
   precisam de largura calculada, redução local de `\tabcolsep`, fonte 10 pt e, se necessário,
   orientação paisagem apenas para o cronograma.
9. **Mesclagens e alinhamentos de células foram perdidos.** Cabeçalhos mesclados, totais e
   alinhamentos específicos do DOCX viraram células vazias ou repetidas, com alinhamento
   `raggedright` uniforme.
10. **Listas com marcadores estão fragmentadas.** Cada item cria um ambiente `itemize` separado.
    Isso introduz espaço vertical excessivo. Os 27 itens devem ser agrupados em listas contínuas.

## Prioridade média

11. **A fonte não é Times New Roman exata.** `mathptmx` é apenas uma fonte semelhante. Para maior
    fidelidade no Overleaf, recomenda-se XeLaTeX com `fontspec` e uma fonte Times compatível,
    normalmente TeX Gyre Termes, ou a fonte institucional disponibilizada legalmente.
12. **Títulos secundários estão em negrito.** O preâmbulo redefine `\ABNTEXsectionfont` como
    negrito, mas os títulos de segundo e terceiro nível do DOCX estão em Times 12 pt sem negrito.
13. **Referências estão com espaçamento 1,5.** A seção deve usar espaçamento simples, alinhamento
    à esquerda e separação consistente entre entradas. O texto atual herda `\OnehalfSpacing`.
14. **Posicionamento rígido de todas as figuras.** O uso de `[H]` nas 19 figuras pode criar
    grandes áreas vazias e quebras ruins. Deve-se permitir `[htbp]`, mantendo legenda, imagem e
    fonte juntas.
15. **Capa e folha de rosto usam espaços verticais fixos.** Elas se aproximam das medidas do DOCX,
    mas valores em pontos são frágeis quando o motor ou a fonte mudam. Precisam ser conferidas no
    PDF final e ajustadas com base na página renderizada.
16. **Tipografia interna das tabelas.** As tabelas herdam corpo 12 e espaçamento 1,5. O DOCX usa
    tratamento mais compacto; é necessário aplicar 10 pt e espaçamento simples localmente.

## Itens que já estão corretos

- Papel A4 e orientação retrato.
- Margens de 3 cm na parte superior e esquerda e 2 cm na parte inferior e direita.
- Corpo principal em 12 pt, justificado, com recuo de 1,25 cm e espaçamento 1,5.
- Estrutura dos sete capítulos e 31 seções/subseções.
- Presença das 19 imagens e das 13 tabelas do DOCX.
- Resumo e abstract em espaçamento simples.
- Compilação LaTeX sem erro fatal.

## Ordem recomendada de correção

1. Remover as três tabelas pré-textuais duplicadas.
2. Separar parte textual, referências e apêndice corretamente.
3. Criar ambiente e lista próprios para quadros.
4. Corrigir legendas e remover fontes duplicadas.
5. Restaurar dimensões originais das 19 imagens.
6. Reconstruir os quadros com 10 pt, mesclagens e larguras adequadas.
7. Agrupar listas de marcadores.
8. Ajustar níveis de títulos, referências e tipografia Times.
9. Compilar e revisar visualmente todas as páginas no Overleaf.
