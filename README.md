## Meta-Unsupervised Learning: Application to Non-Negative Matrix Factorization

Meta-NMF proposes a new approach using the concepts of meta-learning to enhance initialization of NMF in order to improve residual limitations of NMF like initialization sensitivity and generalization ability.

Following is the algorithm that has been implemented in this code:

<figure>
  <img src="https://github.com/user-attachments/assets/7c5ee2c2-dc8f-47f2-a414-bd287cc1f710" alt="">
</figure>

Moreover, it follows this flowchart:
<figure>
  <img src="https://github.com/user-attachments/assets/6bcc6d45-65c3-45cb-a175-2c5988d02b21" alt="">
</figure>

It has been applied to five datasets to validate its applicability and performance:
1. Blobs (synthetic)
2. Coil20
3. Digits
4. Faces
5. Fashion MNIST

Experiments include comparison of the following variants of NMF:
1. NMF - Meta-NMF
2. GNMF - Meta-GNMF
3. NMF NNDSVD - Meta-NMF NNDSVD
4. GNMF NNDSVD - Meta-GNMF NNDSVD

Results show significant improvements in unsupervised clustering performance in terms of NMI. Following is a snapshot of results:

<figure>
  <img src="https://github.com/user-attachments/assets/e70c1c4e-06cb-4e23-aad6-7db282563748" alt="">
</figure>

Furthermore, the ability to extract important facial components is also presented. Following is a snapshot of subjective analysis results:

<figure>
  <img src="https://github.com/user-attachments/assets/d4ebb7c0-c585-4439-8871-ab5df8227009" alt="">
</figure>

## Authors:
1. Ameer Ahmed Khan, Bishop's University, CA
2. Dr. Rachid Hedjam, Bishop's University, CA
3. Dr. Mebarka Allaoui, Bishop's University, CA
4. Dr. Guoqiang Zhong, College of Computer Science and Technology, Ocean University of China, CN
