const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + process.argv[2], { waitUntil: 'networkidle' });
  await p.addStyleTag({content: '@page{size:A4;margin:12mm 10mm} body{font-size:12px} .jp-Notebook{padding:0!important} div.jp-Cell{padding:2px!important} .jp-InputPrompt,.jp-OutputPrompt{min-width:0!important;width:0!important;visibility:hidden} img{max-width:100%!important} .jp-RenderedHTMLCommon table{font-size:11px!important} .jp-RenderedHTMLCommon td,.jp-RenderedHTMLCommon th{font-size:11px!important} pre{white-space:pre-wrap!important;word-break:break-word}'});
  await p.pdf({ path: process.argv[3], format: 'A4', printBackground: true, scale: 0.8 });
  await b.close();
})();
