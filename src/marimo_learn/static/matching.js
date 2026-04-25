// src/help.js
function addHelpButton(container, lang, helpText) {
  const text = helpText[lang] || helpText["en"];
  const popup = document.createElement("div");
  popup.className = "forma-help-popup";
  popup.textContent = text;
  popup.hidden = true;
  const btn = document.createElement("button");
  btn.className = "forma-help-btn";
  btn.textContent = "?";
  btn.setAttribute("aria-label", "Help");
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    popup.hidden = !popup.hidden;
  });
  document.addEventListener("click", () => {
    popup.hidden = true;
  });
  container.append(popup, btn);
}

// src/chota.css
var chota_default = `/*!
 * chota.css v0.9.2 | MIT License | https://github.com/jenil/chota
 */:root{--bg-color:#fff;--bg-secondary-color:#f3f3f6;--color-primary:#14854f;--color-lightGrey:#d2d6dd;--color-grey:#747681;--color-darkGrey:#3f4144;--color-error:#d43939;--color-success:#28bd14;--grid-maxWidth:120rem;--grid-gutter:2rem;--font-size:1.6rem;--font-color:#333;--font-family-sans:-apple-system,"BlinkMacSystemFont","Avenir","Avenir Next","Segoe UI","Roboto","Oxygen","Ubuntu","Cantarell","Fira Sans","Droid Sans","Helvetica Neue",sans-serif;--font-family-mono:monaco,"Consolas","Lucida Console",monospace}html{-webkit-text-size-adjust:100%;-moz-text-size-adjust:100%;-ms-text-size-adjust:100%;text-size-adjust:100%;-webkit-box-sizing:border-box;box-sizing:border-box;font-size:62.5%;line-height:1.15}*,:after,:before{-webkit-box-sizing:inherit;box-sizing:inherit}body{background-color:var(--bg-color);color:var(--font-color);font-family:Segoe UI,Helvetica Neue,sans-serif;font-family:var(--font-family-sans);font-size:var(--font-size);line-height:1.6;margin:0;padding:0}h1,h2,h3,h4,h5,h6{font-weight:500;margin:.35em 0 .7em}h1{font-size:2em}h2{font-size:1.75em}h3{font-size:1.5em}h4{font-size:1.25em}h5{font-size:1em}h6{font-size:.85em}a{color:var(--color-primary);text-decoration:none}a:hover:not(.button){opacity:.75}button{font-family:inherit}p{margin-top:0}blockquote{background-color:var(--bg-secondary-color);border-left:3px solid var(--color-lightGrey);padding:1.5rem 2rem}dl dt{font-weight:700}hr{background-color:var(--color-lightGrey);height:1px;margin:1rem 0}hr,table{border:none}table{border-collapse:collapse;border-spacing:0;text-align:left;width:100%}table.striped tr:nth-of-type(2n){background-color:var(--bg-secondary-color)}td,th{padding:1.2rem .4rem;vertical-align:middle}thead{border-bottom:2px solid var(--color-lightGrey)}tfoot{border-top:2px solid var(--color-lightGrey)}code,kbd,pre,samp,tt{font-family:var(--font-family-mono)}code,kbd{border-radius:4px;color:var(--color-error);font-size:90%;padding:.2em .4em;white-space:pre-wrap}code,kbd,pre{background-color:var(--bg-secondary-color)}pre{font-size:1em;overflow-x:auto;padding:1rem}pre code{background:none;padding:0}abbr[title]{border-bottom:none;text-decoration:underline;-webkit-text-decoration:underline dotted;text-decoration:underline dotted}img{max-width:100%}fieldset{border:1px solid var(--color-lightGrey)}iframe{border:0}.container{margin:0 auto;max-width:var(--grid-maxWidth);padding:0 calc(var(--grid-gutter)/2);width:96%}.row{-webkit-box-direction:normal;-webkit-box-pack:start;-ms-flex-pack:start;display:-webkit-box;display:-ms-flexbox;display:flex;-ms-flex-flow:row wrap;flex-flow:row wrap;justify-content:flex-start;margin-left:calc(var(--grid-gutter)/-2);margin-right:calc(var(--grid-gutter)/-2)}.row,.row.reverse{-webkit-box-orient:horizontal}.row.reverse{-webkit-box-direction:reverse;-ms-flex-direction:row-reverse;flex-direction:row-reverse}.col{-webkit-box-flex:1;-ms-flex:1;flex:1}.col,[class*=" col-"],[class^=col-]{margin:0 calc(var(--grid-gutter)/2) calc(var(--grid-gutter)/2)}.col-1{-ms-flex:0 0 calc(8.33333% - var(--grid-gutter));flex:0 0 calc(8.33333% - var(--grid-gutter));max-width:calc(8.33333% - var(--grid-gutter))}.col-1,.col-2{-webkit-box-flex:0}.col-2{-ms-flex:0 0 calc(16.66667% - var(--grid-gutter));flex:0 0 calc(16.66667% - var(--grid-gutter));max-width:calc(16.66667% - var(--grid-gutter))}.col-3{-ms-flex:0 0 calc(25% - var(--grid-gutter));flex:0 0 calc(25% - var(--grid-gutter));max-width:calc(25% - var(--grid-gutter))}.col-3,.col-4{-webkit-box-flex:0}.col-4{-ms-flex:0 0 calc(33.33333% - var(--grid-gutter));flex:0 0 calc(33.33333% - var(--grid-gutter));max-width:calc(33.33333% - var(--grid-gutter))}.col-5{-ms-flex:0 0 calc(41.66667% - var(--grid-gutter));flex:0 0 calc(41.66667% - var(--grid-gutter));max-width:calc(41.66667% - var(--grid-gutter))}.col-5,.col-6{-webkit-box-flex:0}.col-6{-ms-flex:0 0 calc(50% - var(--grid-gutter));flex:0 0 calc(50% - var(--grid-gutter));max-width:calc(50% - var(--grid-gutter))}.col-7{-ms-flex:0 0 calc(58.33333% - var(--grid-gutter));flex:0 0 calc(58.33333% - var(--grid-gutter));max-width:calc(58.33333% - var(--grid-gutter))}.col-7,.col-8{-webkit-box-flex:0}.col-8{-ms-flex:0 0 calc(66.66667% - var(--grid-gutter));flex:0 0 calc(66.66667% - var(--grid-gutter));max-width:calc(66.66667% - var(--grid-gutter))}.col-9{-ms-flex:0 0 calc(75% - var(--grid-gutter));flex:0 0 calc(75% - var(--grid-gutter));max-width:calc(75% - var(--grid-gutter))}.col-10,.col-9{-webkit-box-flex:0}.col-10{-ms-flex:0 0 calc(83.33333% - var(--grid-gutter));flex:0 0 calc(83.33333% - var(--grid-gutter));max-width:calc(83.33333% - var(--grid-gutter))}.col-11{-ms-flex:0 0 calc(91.66667% - var(--grid-gutter));flex:0 0 calc(91.66667% - var(--grid-gutter));max-width:calc(91.66667% - var(--grid-gutter))}.col-11,.col-12{-webkit-box-flex:0}.col-12{-ms-flex:0 0 calc(100% - var(--grid-gutter));flex:0 0 calc(100% - var(--grid-gutter));max-width:calc(100% - var(--grid-gutter))}@media screen and (max-width:599px){.container{width:100%}.col,[class*=col-],[class^=col-]{-webkit-box-flex:0;-ms-flex:0 1 100%;flex:0 1 100%;max-width:100%}}@media screen and (min-width:900px){.col-1-md{-webkit-box-flex:0;-ms-flex:0 0 calc(8.33333% - var(--grid-gutter));flex:0 0 calc(8.33333% - var(--grid-gutter));max-width:calc(8.33333% - var(--grid-gutter))}.col-2-md{-webkit-box-flex:0;-ms-flex:0 0 calc(16.66667% - var(--grid-gutter));flex:0 0 calc(16.66667% - var(--grid-gutter));max-width:calc(16.66667% - var(--grid-gutter))}.col-3-md{-webkit-box-flex:0;-ms-flex:0 0 calc(25% - var(--grid-gutter));flex:0 0 calc(25% - var(--grid-gutter));max-width:calc(25% - var(--grid-gutter))}.col-4-md{-webkit-box-flex:0;-ms-flex:0 0 calc(33.33333% - var(--grid-gutter));flex:0 0 calc(33.33333% - var(--grid-gutter));max-width:calc(33.33333% - var(--grid-gutter))}.col-5-md{-webkit-box-flex:0;-ms-flex:0 0 calc(41.66667% - var(--grid-gutter));flex:0 0 calc(41.66667% - var(--grid-gutter));max-width:calc(41.66667% - var(--grid-gutter))}.col-6-md{-webkit-box-flex:0;-ms-flex:0 0 calc(50% - var(--grid-gutter));flex:0 0 calc(50% - var(--grid-gutter));max-width:calc(50% - var(--grid-gutter))}.col-7-md{-webkit-box-flex:0;-ms-flex:0 0 calc(58.33333% - var(--grid-gutter));flex:0 0 calc(58.33333% - var(--grid-gutter));max-width:calc(58.33333% - var(--grid-gutter))}.col-8-md{-webkit-box-flex:0;-ms-flex:0 0 calc(66.66667% - var(--grid-gutter));flex:0 0 calc(66.66667% - var(--grid-gutter));max-width:calc(66.66667% - var(--grid-gutter))}.col-9-md{-webkit-box-flex:0;-ms-flex:0 0 calc(75% - var(--grid-gutter));flex:0 0 calc(75% - var(--grid-gutter));max-width:calc(75% - var(--grid-gutter))}.col-10-md{-webkit-box-flex:0;-ms-flex:0 0 calc(83.33333% - var(--grid-gutter));flex:0 0 calc(83.33333% - var(--grid-gutter));max-width:calc(83.33333% - var(--grid-gutter))}.col-11-md{-webkit-box-flex:0;-ms-flex:0 0 calc(91.66667% - var(--grid-gutter));flex:0 0 calc(91.66667% - var(--grid-gutter));max-width:calc(91.66667% - var(--grid-gutter))}.col-12-md{-webkit-box-flex:0;-ms-flex:0 0 calc(100% - var(--grid-gutter));flex:0 0 calc(100% - var(--grid-gutter));max-width:calc(100% - var(--grid-gutter))}}@media screen and (min-width:1200px){.col-1-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(8.33333% - var(--grid-gutter));flex:0 0 calc(8.33333% - var(--grid-gutter));max-width:calc(8.33333% - var(--grid-gutter))}.col-2-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(16.66667% - var(--grid-gutter));flex:0 0 calc(16.66667% - var(--grid-gutter));max-width:calc(16.66667% - var(--grid-gutter))}.col-3-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(25% - var(--grid-gutter));flex:0 0 calc(25% - var(--grid-gutter));max-width:calc(25% - var(--grid-gutter))}.col-4-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(33.33333% - var(--grid-gutter));flex:0 0 calc(33.33333% - var(--grid-gutter));max-width:calc(33.33333% - var(--grid-gutter))}.col-5-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(41.66667% - var(--grid-gutter));flex:0 0 calc(41.66667% - var(--grid-gutter));max-width:calc(41.66667% - var(--grid-gutter))}.col-6-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(50% - var(--grid-gutter));flex:0 0 calc(50% - var(--grid-gutter));max-width:calc(50% - var(--grid-gutter))}.col-7-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(58.33333% - var(--grid-gutter));flex:0 0 calc(58.33333% - var(--grid-gutter));max-width:calc(58.33333% - var(--grid-gutter))}.col-8-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(66.66667% - var(--grid-gutter));flex:0 0 calc(66.66667% - var(--grid-gutter));max-width:calc(66.66667% - var(--grid-gutter))}.col-9-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(75% - var(--grid-gutter));flex:0 0 calc(75% - var(--grid-gutter));max-width:calc(75% - var(--grid-gutter))}.col-10-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(83.33333% - var(--grid-gutter));flex:0 0 calc(83.33333% - var(--grid-gutter));max-width:calc(83.33333% - var(--grid-gutter))}.col-11-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(91.66667% - var(--grid-gutter));flex:0 0 calc(91.66667% - var(--grid-gutter));max-width:calc(91.66667% - var(--grid-gutter))}.col-12-lg{-webkit-box-flex:0;-ms-flex:0 0 calc(100% - var(--grid-gutter));flex:0 0 calc(100% - var(--grid-gutter));max-width:calc(100% - var(--grid-gutter))}}fieldset{padding:.5rem 2rem}legend{font-size:.8em;letter-spacing:.1rem;text-transform:uppercase}input:not([type=checkbox],[type=radio],[type=submit],[type=color],[type=button],[type=reset]),select,textarea,textarea[type=text]{border:1px solid var(--color-lightGrey);border-radius:4px;display:block;font-family:inherit;font-size:1em;padding:.8rem 1rem;-webkit-transition:all .2s ease;transition:all .2s ease;width:100%}select{-webkit-appearance:none;-moz-appearance:none;appearance:none;background:#f3f3f6 no-repeat 100%;background-image:url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='60' height='40' fill='%23555'><polygon points='0,0 60,0 30,40'/></svg>");background-origin:content-box;background-size:1ex}.button,[type=button],[type=reset],[type=submit],button{background:var(--color-lightGrey);border:1px solid transparent;border-radius:4px;color:var(--color-darkGrey);cursor:pointer;display:inline-block;font-size:var(--font-size);line-height:1;padding:1rem 2.5rem;text-align:center;text-decoration:none;-webkit-transform:scale(1);transform:scale(1);-webkit-transition:opacity .2s ease;transition:opacity .2s ease}.button.dark,.button.error,.button.primary,.button.secondary,.button.success,[type=submit]{background-color:#000;background-color:var(--color-primary);color:#fff;z-index:1}.button:hover,[type=button]:hover,[type=reset]:hover,[type=submit]:hover,button:hover{opacity:.8}button:disabled,button:disabled:hover,input:disabled,input:disabled:hover{cursor:not-allowed;opacity:.4}.grouped{display:-webkit-box;display:-ms-flexbox;display:flex}.grouped>:not(:last-child){margin-right:16px}.grouped.gapless>*{border-radius:0!important;margin:0 0 0 -1px!important}.grouped.gapless>:first-child{border-radius:4px 0 0 4px!important;margin:0!important}.grouped.gapless>:last-child{border-radius:0 4px 4px 0!important}input:not([type=checkbox],[type=radio],[type=submit],[type=color],[type=button],[type=reset],:disabled):hover,select:hover,textarea:hover,textarea[type=text]:hover{border-color:var(--color-grey)}input:not([type=checkbox],[type=radio],[type=submit],[type=color],[type=button],[type=reset]):focus,select:focus,textarea:focus,textarea[type=text]:focus{border-color:var(--color-primary);-webkit-box-shadow:0 0 1px var(--color-primary);box-shadow:0 0 1px var(--color-primary);outline:none}input.error:not([type=checkbox],[type=radio],[type=submit],[type=color],[type=button],[type=reset]),textarea.error{border-color:var(--color-error)}input.success:not([type=checkbox],[type=radio],[type=submit],[type=color],[type=button],[type=reset]),textarea.success{border-color:var(--color-success)}[type=checkbox],[type=radio]{height:1.6rem;width:2rem}.button+.button{margin-left:1rem}.button.secondary{background-color:var(--color-grey)}.button.dark{background-color:var(--color-darkGrey)}.button.error{background-color:var(--color-error)}.button.success{background-color:var(--color-success)}.button.outline{background-color:transparent;border-color:var(--color-lightGrey)}.button.outline.primary{border-color:var(--color-primary);color:var(--color-primary)}.button.outline.secondary{border-color:var(--color-grey);color:var(--color-grey)}.button.outline.dark{border-color:var(--color-darkGrey);color:var(--color-darkGrey)}.button.clear{background-color:transparent;border-color:transparent;color:var(--color-primary)}.button.icon{-webkit-box-align:center;-ms-flex-align:center;align-items:center;display:-webkit-inline-box;display:-ms-inline-flexbox;display:inline-flex}.button.icon>img{margin-left:2px}.button.icon-only{padding:1rem}.button:active:not(:disabled),[type=button]:active:not(:disabled),[type=reset]:active:not(:disabled),[type=submit]:active:not(:disabled),button:active:not(:disabled){-webkit-transform:scale(.98);transform:scale(.98)}::-webkit-input-placeholder{color:#bdbfc4}::-moz-placeholder{color:#bdbfc4}:-ms-input-placeholder{color:#bdbfc4}::-ms-input-placeholder{color:#bdbfc4}::placeholder{color:#bdbfc4}.nav{-webkit-box-align:stretch;-ms-flex-align:stretch;align-items:stretch;display:-webkit-box;display:-ms-flexbox;display:flex;min-height:5rem}.nav img{max-height:3rem}.nav-center,.nav-left,.nav-right,.nav>.container{display:-webkit-box;display:-ms-flexbox;display:flex}.nav-center,.nav-left,.nav-right{-webkit-box-flex:1;-ms-flex:1;flex:1}.nav-left{-webkit-box-pack:start;-ms-flex-pack:start;justify-content:flex-start}.nav-right{-webkit-box-pack:end;-ms-flex-pack:end;justify-content:flex-end}.nav-center{-webkit-box-pack:center;-ms-flex-pack:center;justify-content:center}@media screen and (max-width:480px){.nav,.nav>.container{-webkit-box-orient:vertical;-webkit-box-direction:normal;-ms-flex-direction:column;flex-direction:column}.nav-center,.nav-left,.nav-right{-webkit-box-pack:center;-ms-flex-pack:center;-ms-flex-wrap:wrap;flex-wrap:wrap;justify-content:center}}.nav .brand,.nav a{-webkit-box-align:center;-ms-flex-align:center;align-items:center;color:var(--color-darkGrey);display:-webkit-box;display:-ms-flexbox;display:flex;padding:1rem 2rem;text-decoration:none}.nav .active:not(.button),.nav [aria-current=page]:not(.button){color:#000;color:var(--color-primary)}.nav .brand{font-size:1.75em;padding-bottom:0;padding-top:0}.nav .brand img{padding-right:1rem}.nav .button{margin:auto 1rem}.card{background:var(--bg-color);border-radius:4px;-webkit-box-shadow:0 1px 3px var(--color-grey);box-shadow:0 1px 3px var(--color-grey);padding:1rem 2rem}.card p:last-child{margin:0}.card header>*{margin-bottom:1rem;margin-top:0}.tabs{display:-webkit-box;display:-ms-flexbox;display:flex}.tabs a{text-decoration:none}.tabs>.dropdown>summary,.tabs>a{-webkit-box-flex:0;border-bottom:2px solid var(--color-lightGrey);color:var(--color-darkGrey);-ms-flex:0 1 auto;flex:0 1 auto;padding:1rem 2rem;text-align:center}.tabs>a.active,.tabs>a:hover,.tabs>a[aria-current=page]{border-bottom:2px solid var(--color-darkGrey);opacity:1}.tabs>a.active,.tabs>a[aria-current=page]{border-color:var(--color-primary)}.tabs.is-full a{-webkit-box-flex:1;-ms-flex:1 1 auto;flex:1 1 auto}.tag{border:1px solid var(--color-lightGrey);color:var(--color-grey);display:inline-block;letter-spacing:.5px;line-height:1;padding:.5rem;text-transform:uppercase}.tag.is-small{font-size:.75em;padding:.4rem}.tag.is-large{font-size:1.125em;padding:.7rem}.tag+.tag{margin-left:1rem}details.dropdown{display:inline-block;position:relative}details.dropdown>:last-child{left:0;position:absolute;white-space:nowrap}.bg-primary{background-color:var(--color-primary)!important}.bg-light{background-color:var(--color-lightGrey)!important}.bg-dark{background-color:var(--color-darkGrey)!important}.bg-grey{background-color:var(--color-grey)!important}.bg-error{background-color:var(--color-error)!important}.bg-success{background-color:var(--color-success)!important}.bd-primary{border:1px solid var(--color-primary)!important}.bd-light{border:1px solid var(--color-lightGrey)!important}.bd-dark{border:1px solid var(--color-darkGrey)!important}.bd-grey{border:1px solid var(--color-grey)!important}.bd-error{border:1px solid var(--color-error)!important}.bd-success{border:1px solid var(--color-success)!important}.text-primary{color:var(--color-primary)!important}.text-light{color:var(--color-lightGrey)!important}.text-dark{color:var(--color-darkGrey)!important}.text-grey{color:var(--color-grey)!important}.text-error{color:var(--color-error)!important}.text-success{color:var(--color-success)!important}.text-white{color:#fff!important}.pull-right{float:right!important}.pull-left{float:left!important}.text-center{text-align:center}.text-left{text-align:left}.text-right{text-align:right}.text-justify{text-align:justify}.text-uppercase{text-transform:uppercase}.text-lowercase{text-transform:lowercase}.text-capitalize{text-transform:capitalize}.is-full-screen{min-height:100vh;width:100%}.is-full-width{width:100%!important}.is-vertical-align{-webkit-box-align:center;-ms-flex-align:center;align-items:center;display:-webkit-box;display:-ms-flexbox;display:flex}.is-center,.is-horizontal-align{-webkit-box-pack:center;-ms-flex-pack:center;display:-webkit-box;display:-ms-flexbox;display:flex;justify-content:center}.is-center{-webkit-box-align:center;-ms-flex-align:center;align-items:center}.is-right{-webkit-box-pack:end;-ms-flex-pack:end;justify-content:flex-end}.is-left,.is-right{-webkit-box-align:center;-ms-flex-align:center;align-items:center;display:-webkit-box;display:-ms-flexbox;display:flex}.is-left{-webkit-box-pack:start;-ms-flex-pack:start;justify-content:flex-start}.is-fixed{position:fixed;width:100%}.is-paddingless{padding:0!important}.is-marginless{margin:0!important}.is-pointer{cursor:pointer!important}.is-rounded{border-radius:100%}.clearfix{clear:both;content:"";display:table}.is-hidden{display:none!important}@media screen and (max-width:599px){.hide-xs{display:none!important}}@media screen and (min-width:600px) and (max-width:899px){.hide-sm{display:none!important}}@media screen and (min-width:900px) and (max-width:1199px){.hide-md{display:none!important}}@media screen and (min-width:1200px){.hide-lg{display:none!important}}@media print{.hide-pr{display:none!important}}
`;

// src/forma.css
var forma_default = '/*\n * Forma widget styles \u2014 loaded alongside chota.css.\n *\n * To customize all widgets, override a small number of vars on :root:\n *\n *   Chota vars (affect chota UI and forma widgets):\n *     --color-primary      accent color for buttons, borders, highlights\n *     --bg-color           page/widget background\n *     --bg-secondary-color muted background, secondary buttons\n *     --color-lightGrey    border color\n *     --color-grey         secondary text\n *     --color-error        danger/error color\n *     --color-success      correct/success color\n *\n *   Forma-only vars (override forma widgets only, take precedence over chota):\n *     --forma-primary      (default: --color-primary)\n *     --forma-bg           (default: --bg-color)\n *     --forma-border       (default: --color-lightGrey)\n *     --forma-radius       corner rounding (default: 4px)\n *     --forma-accent       hover/highlight tint (default: --bg-secondary-color)\n */\n\n.forma {\n  /* Internal aliases: forma var \u2192 chota var \u2192 hardcoded default.\n   * Widget rules use var(--pri), var(--bg), etc. for brevity. */\n  --pri:  var(--forma-primary,       var(--color-primary,       #14854f));\n  --pfg:  var(--forma-primary-fg,    #fff);\n  --bg:   var(--forma-bg,            var(--bg-color,            #fff));\n  --txt:  var(--forma-text,          var(--font-color,          #333));\n  --bdr:  var(--forma-border,        var(--color-lightGrey,     #d2d6dd));\n  --mut:  var(--forma-muted,         var(--bg-secondary-color,  #f3f3f6));\n  --sec:  var(--forma-secondary,     var(--bg-secondary-color,  #f3f3f6));\n  --dim:  var(--forma-text-muted,    var(--color-grey,          #747681));\n  --err:  var(--forma-danger,        var(--color-error,         #d43939));\n  --ok:   var(--forma-correct-color, var(--color-success,       #28bd14));\n  --acc:  var(--forma-accent,           var(--bg-secondary-color,  #f3f3f6));\n  --rad:  var(--forma-radius,           4px);\n  /* State colors */\n  --cbg:  var(--forma-correct-bg,       #d4edda);\n  --ibg:  var(--forma-incorrect-bg,     #f8d7da);\n  --drbg: var(--forma-draggable-bg,     var(--bg-secondary-color,  #f3f3f6));\n  --drbd: var(--forma-draggable-border, var(--color-primary,        #14854f));\n  --dpbg: var(--forma-drop-bg,          var(--bg-secondary-color,  #f3f3f6));\n  --dpbd: var(--forma-drop-border,      var(--color-darkGrey,       #3f4144));\n\n  position: relative;\n  padding: 12px;\n  border: 1px solid var(--bdr);\n  border-radius: var(--rad);\n}\n\n/* Dark mode overrides */\n.dark .forma, .dark-theme .forma, [data-theme="dark"] .forma {\n  --forma-bg:              #1e1e2e;\n  --forma-text:            #cdd6f4;\n  --forma-text-muted:      #9399b2;\n  --forma-border:          #45475a;\n  --forma-accent:          #313244;\n  --forma-muted:           #181825;\n  --forma-secondary:       #313244;\n  --forma-correct-bg:      #14532d;\n  --forma-correct-color:   #4ade80;\n  --forma-incorrect-bg:    #450a0a;\n  --forma-draggable-bg:    #431407;\n  --forma-draggable-border:#ea580c;\n  --forma-drop-bg:         #422006;\n  --forma-drop-border:     #ca8a04;\n}\n\n.forma-question     { font-size: 1.1em; margin-bottom: 12px; }\n.forma-instructions { font-size: 0.9em; color: var(--dim); margin-bottom: 12px; }\n\n/* Buttons */\n.forma-btn {\n  padding: 10px 20px; border: none; border-radius: var(--rad);\n  font: inherit; cursor: pointer;\n}\n.forma-btn-primary   { background: var(--pri); color: var(--pfg); }\n.forma-btn-secondary { background: var(--sec); color: var(--txt); border: 1px solid var(--bdr); }\n.forma-btn-danger    { background: var(--err); color: var(--pfg); }\n.forma-btn-primary:hover:not(:disabled),\n.forma-btn-danger:hover:not(:disabled)    { opacity: 0.9; }\n.forma-btn-secondary:hover:not(:disabled) { opacity: 0.85; }\n.forma-btn:disabled { opacity: 0.45; cursor: not-allowed; }\n\n/* Feedback */\n.forma-feedback  { margin-top: 12px; }\n.forma-correct   { color: var(--ok); }\n.forma-incorrect { color: var(--err); }\n\n/* Explanation block */\n.forma-explanation {\n  padding: 10px; margin-top: 10px;\n  background: var(--mut); border-left: 4px solid var(--pri); border-radius: var(--rad);\n}\n\n/* Shared: padded bordered boxes used by multiple widgets */\n.forma-option,\n.forma-item-fixed,\n.forma-drop-zone,\n.forma-ordering-item,\n.forma-item-draggable {\n  padding: 10px;\n  border: 2px solid var(--bdr);\n  border-radius: var(--rad);\n}\n\n/* Shared: correct/incorrect state applied across widget item types */\n.forma-option.forma-correct,\n.forma-item-fixed.forma-correct,\n.forma-drop-zone.forma-correct,\n.forma-ordering-item.forma-correct { background: var(--cbg); border-color: var(--ok); }\n\n.forma-option.forma-incorrect,\n.forma-item-fixed.forma-incorrect,\n.forma-drop-zone.forma-incorrect,\n.forma-ordering-item.forma-incorrect { background: var(--ibg); border-color: var(--err); }\n\n/* Multiple choice */\n.forma-options { margin-bottom: 12px; }\n.forma-option  { margin: 6px 0; cursor: pointer; transition: background .2s, border-color .2s; }\n.forma-option:hover:not(.forma-answered) { background: var(--acc); border-color: var(--pri); }\n.forma-option.forma-answered { cursor: not-allowed; }\n.forma-option.forma-faded    { opacity: .5; }\n\n/* Matching */\n.forma-matching-three-col {\n  display: grid; grid-template-columns: 1fr 1fr 1fr;\n  column-gap: 16px; row-gap: 8px; margin-bottom: 16px; align-items: stretch;\n}\n.forma-item-fixed { background: var(--acc); border-color: var(--pri); }\n\n/* Drop zones */\n.forma-drop-zone {\n  min-height: 40px; background: var(--mut); border-style: dashed;\n  text-align: center; color: var(--dim); font-style: italic;\n  cursor: pointer; transition: all .2s;\n}\n.forma-drop-zone.forma-filled { background: var(--sec); border-style: solid; color: inherit; font-style: normal; text-align: left; }\n.forma-drop-zone.forma-drop-target,\n.forma-label-drop-zone.forma-drop-target { background: var(--dpbg); border-color: var(--dpbd); }\n\n.forma-item-draggable { background: var(--drbg); border-color: var(--drbd); }\n\n/* Ordering */\n.forma-ordering-items { margin-bottom: 16px; }\n.forma-ordering-item  { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }\n.forma-ordering-item.forma-drop-target { background: var(--dpbg); border-style: dashed; }\n.forma-ordering-text  { flex: 1; }\n\n/* Shared: drag cursor and dragging state */\n.forma-ordering-item,\n.forma-item-draggable,\n.forma-label-num,\n.forma-label-badge     { cursor: grab; }\n.forma-ordering-item:active,\n.forma-item-draggable:active,\n.forma-label-num:active,\n.forma-label-badge:active { cursor: grabbing; }\n.forma-ordering-item.forma-dragging,\n.forma-item-draggable.forma-dragging,\n.forma-label-num.forma-dragging,\n.forma-label-badge.forma-dragging { opacity: .5; }\n\n/* Shared: circular position/label badges */\n.forma-position,\n.forma-label-num,\n.forma-label-badge {\n  background: var(--pri); color: var(--pfg);\n  border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0;\n}\n.forma-position,\n.forma-label-num  { min-width: 30px; height: 30px; }\n.forma-label-badge { min-width: 24px; height: 24px; font-size: 0.8em; }\n.forma-label-badge.forma-correct   { background: var(--ok);  color: white; }\n.forma-label-badge.forma-incorrect { background: var(--err); color: white; }\n\n/* Labeling */\n.forma-labeling-area   { display: grid; grid-template-columns: 1fr 2fr; gap: 16px; margin-bottom: 16px; }\n.forma-labeling-labels,\n.forma-labeling-text   { display: flex; flex-direction: column; gap: 10px; }\n.forma-labeling-title  { color: var(--dim); margin-bottom: 6px; }\n.forma-label-item      { display: flex; align-items: center; gap: 8px; padding: 6px; border-radius: var(--rad); }\n.forma-text-lines      { display: flex; flex-direction: column; gap: 8px; }\n.forma-text-line       { display: flex; align-items: flex-start; gap: 8px; }\n.forma-label-drop-zone {\n  min-width: 60px; min-height: 30px; padding: 4px;\n  background: var(--mut); border: 2px dashed var(--bdr); border-radius: var(--rad);\n  display: flex; flex-wrap: wrap; gap: 4px; align-content: flex-start; flex-shrink: 0;\n}\n.forma-text-content { flex: 1; padding: 4px 0; line-height: 1.5; }\n\n/* Flashcard */\n.forma-card {\n  min-height: 105px; padding: 24px 15px; border-radius: var(--rad);\n  display: flex; align-items: center; justify-content: center;\n  font-size: 1.2em; text-align: center; margin-bottom: 12px; transition: all .3s;\n}\n.forma-card-front { background: var(--acc); border: 2px solid var(--pri); }\n.forma-card-back  { background: var(--mut); border: 2px solid var(--bdr); }\n.forma-rating-btns   { display: flex; gap: 8px; margin-bottom: 12px; }\n.forma-progress      { height: 6px; background: var(--mut); border-radius: 4px; margin-bottom: 10px; }\n.forma-progress-fill { height: 100%; background: var(--pri); border-radius: 4px; transition: width .3s; }\n\n/* Concept map */\n.forma-cm-workspace {\n  position: relative; height: 360px;\n  border: 1px solid var(--bdr); border-radius: var(--rad);\n  margin-bottom: 12px; overflow: hidden; background: var(--mut);\n}\n.forma-cm-svg  { position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: visible; }\n.forma-cm-node {\n  position: absolute; transform: translate(-50%, -50%);\n  padding: 8px 14px; background: var(--bg); border: 2px solid var(--pri);\n  border-radius: var(--rad); cursor: move; user-select: none; z-index: 1; white-space: nowrap;\n}\n.forma-cm-node.forma-cm-pending { box-shadow: 0 0 0 3px var(--pri); background: var(--acc); }\n.forma-cm-terms { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }\n.forma-cm-term {\n  padding: 5px 14px; border: 2px solid var(--bdr); border-radius: 20px;\n  background: var(--mut); cursor: pointer; font: inherit; font-size: 0.9em; transition: all .15s;\n}\n.forma-cm-term:hover:not(:disabled)    { border-color: var(--pri); background: var(--acc); }\n.forma-cm-term.forma-cm-term-selected { background: var(--pri); color: var(--pfg); border-color: var(--pri); }\n.forma-cm-term:disabled               { opacity: 0.45; cursor: not-allowed; }\n.forma-cm-edge-list { display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px; }\n.forma-cm-edge-row  {\n  display: flex; align-items: center; gap: 6px;\n  padding: 4px 8px; background: var(--mut); border-radius: var(--rad); font-size: 0.9em;\n}\n.forma-cm-edge-label  { padding: 2px 8px; background: var(--acc); border-radius: 10px; font-size: 0.85em; }\n.forma-cm-edge-remove { margin-left: auto; background: none; border: none; cursor: pointer; color: var(--dim); padding: 2px 4px; border-radius: 4px; }\n.forma-cm-edge-remove:hover { color: var(--err); background: var(--ibg); }\n\n/* Numeric entry */\n.forma-numeric-row   { display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }\n.forma-numeric-input {\n  padding: 8px 12px; border: 2px solid var(--bdr); border-radius: var(--rad);\n  font: inherit; font-size: 1em; width: 10em;\n}\n.forma-numeric-input:disabled { opacity: 0.6; cursor: not-allowed; }\n\n/* Shared: monospace display blocks */\n.forma-code,\n.forma-output {\n  padding: 12px 16px;\n  background: var(--mut); border-radius: var(--rad);\n  font-family: monospace; font-size: 0.9em; white-space: pre; overflow-x: auto;\n}\n.forma-code   { border: 1px solid var(--bdr); margin-bottom: 14px; }\n.forma-output { border-left: 4px solid var(--pri); margin-top: 10px; }\n.forma-reveal-btn { margin-top: 12px; }\n\n/* Help button and popup */\n.forma-help-btn {\n  position: absolute; top: 8px; right: 8px;\n  width: 22px; height: 22px; border-radius: 50%;\n  border: 1.5px solid var(--bdr); background: var(--mut); color: var(--dim);\n  font: 0.8em/1 inherit; cursor: pointer;\n  display: flex; align-items: center; justify-content: center; padding: 0; z-index: 10;\n}\n.forma-help-btn:hover { background: var(--acc); border-color: var(--pri); color: var(--txt); }\n.forma-help-popup {\n  position: absolute; top: calc(22px + 12px); right: 0;\n  min-width: 200px; max-width: 320px; padding: 10px 14px;\n  background: var(--bg); border: 1px solid var(--bdr); border-radius: var(--rad);\n  font-size: 0.9em; line-height: 1.5; z-index: 20;\n  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);\n}\n';

// src/utils.js
function mk(tag, cls, txt) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (txt !== void 0) el.textContent = txt;
  return el;
}
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
var DEFAULT_MATH_DELIMITERS = [{ left: "$", right: "$", display: false }];
function renderMath(el, delimiters) {
  window.renderMathInElement?.(el, { delimiters: delimiters ?? DEFAULT_MATH_DELIMITERS });
}
function initWidget(el, question) {
  const s = mk("style");
  s.textContent = chota_default + forma_default;
  el.appendChild(s);
  const container = mk("div", "forma");
  if (question) container.appendChild(mk("div", "forma-question", question));
  return container;
}
function setupDropZone(zone, onDrop, isSubmitted) {
  zone.addEventListener("dragover", (e) => {
    if (isSubmitted()) return;
    e.preventDefault();
    zone.classList.add("forma-drop-target");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("forma-drop-target"));
  zone.addEventListener("drop", (e) => {
    if (isSubmitted()) return;
    e.preventDefault();
    zone.classList.remove("forma-drop-target");
    onDrop(e);
  });
}
function createSubmitRow(submitLabel) {
  const btnRow = mk("div");
  const submitBtn = mk("button", "forma-btn forma-btn-primary", submitLabel);
  submitBtn.style.marginRight = "12px";
  const tryAgainBtn = mk("button", "forma-btn forma-btn-secondary", "Try Again");
  tryAgainBtn.style.display = "none";
  return { btnRow, submitBtn, tryAgainBtn };
}

// src/matching.js
var HELP_TEXT = {
  en: "Drag items from the right column and drop them into the matching slots in the middle column. Click a placed item to remove it and try again. All items must be matched before you can check your answers. Click Try Again to reset and retry.",
  fr: "Faites glisser les \xE9l\xE9ments de la colonne droite et d\xE9posez-les dans les emplacements correspondants de la colonne centrale. Cliquez sur un \xE9l\xE9ment plac\xE9 pour le retirer. Tous les \xE9l\xE9ments doivent \xEAtre appari\xE9s avant de v\xE9rifier. Cliquez sur R\xE9essayer pour recommencer.",
  es: "Arrastre los elementos de la columna derecha y su\xE9ltelos en las ranuras correspondientes de la columna central. Haga clic en un elemento colocado para eliminarlo. Todos los elementos deben estar emparejados antes de verificar. Haga clic en Reintentar para volver a intentarlo."
};
function render({ model, el }) {
  const container = initWidget(el, model.get("question"));
  container.appendChild(mk("div", "forma-instructions", "Drag labels from the right column to match with items on the left:"));
  const left = model.get("left"), right = model.get("right");
  const correctMap = Object.fromEntries(Object.entries(model.get("correct_matches")).map(([k, v]) => [+k, +v]));
  const matches = {};
  let submitted = false;
  const grid = mk("div", "forma-matching-three-col");
  const leftItems = left.map((text) => mk("div", "forma-item-fixed", text));
  const zones = left.map((_, li) => {
    const z = mk("div", "forma-drop-zone", "(drop here)");
    setupDropZone(z, (e) => {
      const ri = parseInt(e.dataTransfer.getData("text/plain"));
      z.textContent = right[ri];
      z.className = "forma-drop-zone forma-filled";
      renderMath(z, model.get("math_delimiters"));
      matches[li] = ri;
      sync();
      z.addEventListener("click", () => {
        if (submitted) return;
        z.textContent = "(drop here)";
        z.className = "forma-drop-zone";
        delete matches[li];
        sync();
      });
    }, () => submitted);
    return z;
  });
  const rightItems = right.map((text, i) => {
    const d = mk("div", "forma-item-draggable", text);
    d.draggable = true;
    d.addEventListener("dragstart", (e) => {
      if (submitted) return;
      d.classList.add("forma-dragging");
      e.dataTransfer.effectAllowed = "copy";
      e.dataTransfer.setData("text/plain", i);
    });
    d.addEventListener("dragend", () => d.classList.remove("forma-dragging"));
    return d;
  });
  left.forEach((_, i) => grid.append(leftItems[i], zones[i], rightItems[i]));
  container.appendChild(grid);
  const { btnRow, submitBtn, tryAgainBtn } = createSubmitRow("Check Answers");
  btnRow.style.marginBottom = "16px";
  submitBtn.addEventListener("click", () => {
    if (submitted) return;
    if (Object.keys(matches).length !== left.length) {
      alert("Please match all items before checking answers.");
      return;
    }
    submitted = true;
    submitBtn.disabled = true;
    rightItems.forEach((d) => {
      d.draggable = false;
      d.style.cssText = "cursor:default;opacity:.5";
    });
    let score = 0;
    zones.forEach((z, li) => {
      const ok = matches[li] === correctMap[li];
      if (ok) score++;
      leftItems[li].classList.add(ok ? "forma-correct" : "forma-incorrect");
      z.classList.add(ok ? "forma-correct" : "forma-incorrect");
      z.appendChild(mk("span", ok ? "forma-correct" : "forma-incorrect", ok ? " \u2713" : " \u2717"));
    });
    container.appendChild(mk("div", `forma-feedback ${score === left.length ? "forma-correct" : "forma-incorrect"}`, `Score: ${score}/${left.length} correct`));
    tryAgainBtn.style.display = "inline-block";
    model.set("value", { matches, correct: score === left.length, score, total: left.length });
    model.save_changes();
  });
  tryAgainBtn.addEventListener("click", () => {
    submitted = false;
    submitBtn.disabled = false;
    tryAgainBtn.style.display = "none";
    Object.keys(matches).forEach((k) => delete matches[k]);
    zones.forEach((z) => {
      z.textContent = "(drop here)";
      z.className = "forma-drop-zone";
    });
    leftItems.forEach((li) => li.classList.remove("forma-correct", "forma-incorrect"));
    rightItems.forEach((d) => {
      d.draggable = true;
      d.style.cssText = "";
    });
    const fb = container.querySelector(".forma-feedback");
    if (fb) fb.remove();
    sync();
  });
  btnRow.append(submitBtn, tryAgainBtn);
  container.appendChild(btnRow);
  addHelpButton(container, model.get("lang"), HELP_TEXT);
  el.appendChild(container);
  function sync() {
    model.set("value", { matches, correct: false, score: 0, total: left.length });
    model.save_changes();
  }
}
function parseHTML(div) {
  const question = div.querySelector("p")?.textContent.trim() ?? "";
  const rows = [...div.querySelectorAll("tr")].filter(
    (r) => r.closest("thead") === null && r.querySelector("th") === null
  );
  const left = rows.map((r) => r.cells[0].textContent.trim());
  const rightOrdered = rows.map((r) => r.cells[1].textContent.trim());
  const indices = rightOrdered.map((_, i) => i);
  shuffle(indices);
  const right = indices.map((i) => rightOrdered[i]);
  const correct_matches = Object.fromEntries(left.map((_, i) => [i, indices.indexOf(i)]));
  const raw = div.dataset.mathDelimiters;
  const math_delimiters = raw ? JSON.parse(raw) : DEFAULT_MATH_DELIMITERS;
  return { question, left, right, correct_matches, lang: div.dataset.lang ?? "en", math_delimiters };
}
var matching_default = { render };
export {
  matching_default as default,
  parseHTML
};
//# sourceMappingURL=matching.js.map
