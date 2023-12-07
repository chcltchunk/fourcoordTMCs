# in DOI: 10.1039/c7sc01247k cutoff is <= 1 (gives 242 less datapoints)
S2_CUTOFF = 1.5
# SSE lower than -110 kcal/mol is not reasonable for this dataset
SSE_CUTOFF = -110


# define properties we'd like to keep in the dataset throughout the complete workflow
# TODO(ralf): should we have the formal charge in the final dataset?
column_list = ['metal', 'ox', 'ligstr', 'complex.size']  # , 'charge'
sse_colum_list = ['geom.ls', 'geom.hs', 'ls.spin', 'hs.spin', 'b3lyp.energy.ls (Ha)', 'b3lyp.energy.hs (Ha)', 'b3lyp.sse (kcal/mol)']
