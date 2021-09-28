(TeX-add-style-hook
 "romano-biorxiv-v1"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "ws-procs11x85"
    "ws-procs11x8510"
    "ws-procs-thm"
    "algorithm"
    "algpseudocode")
   (LaTeX-add-labels
    "introduction"
    "fig:2"
    "methods-nc"
    "alg:1"
    "fig:3"
    "fig:4"
    "fig:5"
    "GCNN"
    "eq:a1"
    "NC"
    "eq:b1")
   (LaTeX-add-bibliographies
    "psb-gnn"))
 :latex)

