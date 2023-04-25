jupyter nbconvert --to notebook --execute notebook/SRBM_unsup_gen.ipynb
mv notebook/SRBM_unsup_gen.nbconvert.ipynb notebook/SRBM_unsup_gen.ipynb
nbstripout notebook/SRBM_unsup_gen.ipynb

jupyter nbconvert --to notebook --execute notebook/SRBM_sup_gen_scalar_field.ipynb
mv notebook/SRBM_sup_gen_scalar_field.nbconvert.ipynb notebook/SRBM_sup_gen_scalar_field.ipynb
nbstripout notebook/SRBM_sup_gen_scalar_field.ipynb

jupyter nbconvert --to notebook --execute notebook/SRBM_sup_gen_MNIST.ipynb
mv notebook/SRBM_sup_gen_MNIST.nbconvert.ipynb notebook/SRBM_sup_gen_MNIST.ipynb
nbstripout notebook/SRBM_sup_gen_MNIST.ipynb

rm ./notebook/*.npz

pytest -v
rm *.npz
