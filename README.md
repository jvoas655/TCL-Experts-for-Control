# Topic Contrastive Language (or TCL) Experts for Control
A method to produce topically stable and coherent long form text generation through the use of dual contrastive topic experts. 

We explore the use of contrasting expert lan- guage models for the use of topical control on language generation. As a result, we exam- ine methods to extract linguistically significant meaning from self-organizing structures trained on a large corpus of Wikipedia-drawn text sum- maries and their corresponding categories. Our findings show extracting such meaning is dif- ficult and needs further work, with the con- trastive experts expressing opposing probability adjustments that simply cancel each other out. Further, when using only our positive expert as opposed to the contrastive experts, we observe metric improvements for our in-domain data but a failure to match pre-prompting baselines on out-of-domain data. Our work is a novel ex- ploration of self-organizing structures for this purpose and presents a starting point for such methods going forward.

Our repository includes a codebase to replicate this model as well as our final paper.
