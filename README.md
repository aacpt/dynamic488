# dynamic488

Circuit-level simulation and decoding for ancilla-free *dynamic* realisations of the CSS 4.8.8 (square-octagon) Floquet code, compared against standard and pipelined ancilla-based syndrome extraction. 

It builds the syndrome extraction circuits with Stim, simulates them under circuit-level depolarising noise, decodes with MWPM and BP+matching, and produces threshold and sub-threshold data.

From the paper *The dynamic 4.8.8 Floquet code*. 