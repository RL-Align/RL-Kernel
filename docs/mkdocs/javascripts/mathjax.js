window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  },
  startup: {
    typeset: false,
    ready() {
      MathJax.startup.defaultReady();

      // Wait for startup and serialize typesetting across instant navigation.
      let pending = MathJax.startup.promise;
      document$.subscribe(() => {
        pending = pending.then(() => {
          MathJax.startup.output.clearCache();
          MathJax.typesetClear();
          MathJax.texReset();
          return MathJax.typesetPromise();
        }).catch(error => {
          console.error("MathJax typesetting failed:", error);
        });
      });
    }
  }
};
