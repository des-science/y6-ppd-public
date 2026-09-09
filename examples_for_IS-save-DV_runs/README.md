As input for PPD calculations, we need a chain where we have saved
data vector theory predictions.

This directory contains some examples of how you would set that up
by running the cosmosis importance sampler on  a previously run cosmsis chain.

Effectively we're using the importance sampler as a way to get cosmosis
to redo theory calculations at each sample in the original chain. We're
NOT going to use the posterior update that comes from the IS run,
(as we would in a standard application of IS),
we're just running a theory calculation for each sample and saving
the data vectory theory predictions as extra ouptut. 
