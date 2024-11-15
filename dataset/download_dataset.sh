#!/bin/bash

# Set up repository URL and branch name
REPO_URL="https://github.com/yumoxu/stocknet-dataset"
BRANCH="master"  # Adjust the branch name if it's different

# Create a temporary directory for the sparse checkout
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR" || exit

# Initialize a sparse Git repository
git init
git remote add origin "$REPO_URL"
git config core.sparseCheckout true

# Specify the folders to download
echo "price/" >> .git/info/sparse-checkout
echo "tweet/" >> .git/info/sparse-checkout

# Fetch only the specified folders
git pull origin "$BRANCH"

# Move the downloaded folders to the current directory
mv price tweet "$OLDPWD"

# Clean up the temporary directory
cd "$OLDPWD" || exit
rm -rf "$TEMP_DIR"

echo "Downloaded 'price' and 'tweet' folders to the current directory."