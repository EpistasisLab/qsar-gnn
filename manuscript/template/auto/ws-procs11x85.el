(TeX-add-style-hook
 "ws-procs11x85"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("ws-procs11x85" "square")))
   (TeX-run-style-hooks
    "latex2e"
    "ws-procs11x8510"
    "ws-procs-thm")
   (TeX-add-symbols
    '("keyw" 1)
    "Examplefont"
    "Exampleheadfont"
    "pc"
    "p")
   (LaTeX-add-labels
    "aba:sec1"
    "aba:theo"
    "aba:the1"
    "aba:the2"
    "aba:eq1"
    "aba:appeq2"
    "aba:appeq3"
    "aba:tbl1"
    "tblabel"
    "aba:fig1"
    "aba:fig2"
    "aba:tbl3"
    "app:a1"
    "aba:app1"
    "aba:app2")
   (LaTeX-add-bibitems
    "jarl88"
    "lamp94"
    "ams04"
    "best03")
   (LaTeX-add-environments
    "example")
   (LaTeX-add-bibliographies
    "ws-pro-sample"))
 :latex)

