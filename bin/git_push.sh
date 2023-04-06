bash ./bin/run_check.sh
bash ./bin/create_env.sh

git add .
git commit -m "$@"
git push origin main
