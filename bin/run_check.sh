jupyter nbconvert --to notebook --execute notebook/SRBM_supervised.ipynb
mv notebook/SRBM_supervised.nbconvert.ipynb notebook/SRBM_supervised.ipynb
nbstripout notebook/SRBM_supervised.ipynb

jupyter nbconvert --to notebook --execute notebook/SRBM_unsupervised.ipynb
mv notebook/SRBM_unsupervised.nbconvert.ipynb notebook/SRBM_unsupervised.ipynb
nbstripout notebook/SRBM_unsupervised.ipynb

jupyter nbconvert --to notebook --execute notebook/SRBM_sup_gen_MNIST.ipynb
mv notebook/SRBM_sup_gen_MNIST.nbconvert.ipynb notebook/SRBM_sup_gen_MNIST.ipynb
nbstripout notebook/SRBM_sup_gen_MNIST.ipynb

rm ./notebook/*.npz

pytest -v
rm *.npz
