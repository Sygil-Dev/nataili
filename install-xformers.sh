if [ ! -f xformers-0.0.14.dev0-cp38-cp38-linux_x86_64.whl ]; then
  wget https://github.com/ninele7/xfromers_builds/releases/download/3352937371/xformers-0.0.14.dev0-cp38-cp38-linux_x86_64.whl
fi

./runtime.sh pip install xformers-0.0.14.dev0-cp38-cp38-linux_x86_64.whl