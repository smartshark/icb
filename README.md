# Issue Classification Benchmark (ICB)
This project tries to replicate the current state-of-the-art of the classification of issues mined from an issue 
tracking system. The goal is to classify issues into issues describing a bug and others, so that approaches that use 
these information (e.g., for defect prediction) have better data.

That the misclassification of issues is a rather large threat to the validity was shown by 
Herzig et al.: "It's not a bug, it's a feature: how misclassification impacts bug prediction" 
(https://dl.acm.org/citation.cfm?id=2486840)

An example of how to use this library is shown in the main.py.

**IMPORTANT**: If you want to use the approach by Terdchanakul et al., you need to install the 
ngweight project (https://github.com/iwnsew/ngweight).

Currently implemented approaches (sorted by Date published)
----------------------------------------------------------

Antoniol et al. (2008): "Is it a bug or an enhancement?: a text-based approach to classify change requests." CASCON. 
DOI: https://doi.org/10.1145/1463788.1463819

Pingclasai et al. (2013): "Classifying Bug Reports to Bugs and Other Requests Using Topic Modeling". 
In: Proceedings of the 20th Asia-Pacific Software Engineering Conference (APSEC). 
DOI: https://doi.org/10.1109/APSEC.2013.105

Limsettho et al. (2014): "Comparing hierarchical dirichlet process with latent dirichlet allocation in bug report 
multiclass classification". In: Proceedings of the 15th IEEE/ACIS International Conference on Software Engineering, 
Artificial Intelligence, Networking and Parallel/Distributed Computing (SNPD)
DOI: https://doi.org/10.1109/SNPD.2014.6888695

Chawla et al. (2015): "An automated approach for bug categorization using fuzzy logic." In: Proceedings of the 
8th India Software Engineering Conference.
DOI: https://doi.org/10.1145/2723742.2723751

Terdchanakul et al. (2017): "Bug or Not? Bug Report Classification Using N-Gram IDF". In: Proceedings of the 
33th IEEE International Conference on Software Maintenance and Evolution
DOI: https://doi.org/10.1109/ICSME.2017.14

**IMPORTANT**: If you want to use the approach by Terdchanakul et al., you need to install the 
ngweight project (https://github.com/iwnsew/ngweight).

Palacio et al. (2018): "Learning to Identify Security-Related Issues Using
Convolutional Neural Networks". In: Proceedings of the 2019 IEEE International Conference on Software Maintenance
and Evolution (ICSME)
DOI: 10.1109/ICSME.2019.00024

Pandey et al. (2018): "Automated Classification of Issue Reports from a Software Issue Tracker". In: 
Progress in Intelligent Computing Techniques: Theory, Practice, and Applications.
DOI: https://doi.org/10.1007/978-981-10-3373-5_42

Qin et al. (2018): "Classifying Bug Reports into Bugs and Non-bugs Using LSTM". In: Proceedings of the 
10th Asia-Pacific Symposium on Internetware
DOI: https://doi.org/10.1145/3275219.3275239

Kallis et al. (2019): "Ticket Tagger: Machine Learning Driven Issue Classification." In: 
Proceedings of the 35th IEEE International Conference on Software Maintenance and Evolution
DOI: (not yet assigned)

Otoom et al. (2019): "Automated Classification of Software Bug Reports." In: Proceedings of the 9th International 
Conference on Information Communication and Management.
DOI: https://doi.org/10.1145/3357419.3357424

