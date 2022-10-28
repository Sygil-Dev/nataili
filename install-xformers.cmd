@echo off
cd /d %~dp0

IF not exist "xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl" (
    curl -L -o "xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl" "https://github.com/ninele7/xfromers_builds/releases/download/3352937371/xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl"
)

call runtime pip install xformers-0.0.14.dev0-cp38-cp38-win_amd64.whl
