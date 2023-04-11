jupyter nbconvert --to notebook --execute notebook/SRBM_unsup_gen.ipynb
mv notebook/SRBM_unsup_gen.nbconvert.ipynb notebook/SRBM_unsup_gen.ipynb
nbstripout notebook/SRBM_unsup_gen.ipynb
rm ./notebook/*.npz

pytest -v
rm *.npz
