#!/bin/bash

#update gitignore_bigfiles
find . -size +30M -not -path "./.git*"| sed 's|^\./||g' | cat > .gitignore_bigfiles

# create gitignore as concat of gitingore_static and gitignore_bigfiles
cat .gitignore_static .gitignore_bigfiles > .gitignore

# print content of .gitignore_bigfiles
cat .gitignore_bigfiles
