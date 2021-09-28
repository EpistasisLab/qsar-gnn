(TeX-add-style-hook
 "r1"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("babel" "english")))
   (TeX-run-style-hooks
    "latex2e"
    "ibilttr"
    "ibilttr10"
    "inputenc"
    "babel"
    "csquotes"))
 :latex)

