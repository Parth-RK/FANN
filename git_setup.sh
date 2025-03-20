#!/bin/bash

# --------------------------------------------------------------
# This script helps configure Git with your username, email, 
# and access token for authentication, along with optional 
# credential caching and SSH key setup for GitHub/GitLab/other Git providers.
# --------------------------------------------------------------

# Check if Git is installed
if ! command -v git &> /dev/null
then
    echo "Git is not installed. Please install Git first."
    exit 1
fi

# Prompt user to enter their Git username
echo "Enter your Git username:"
read git_username

# Prompt user to enter their Git email address
echo "Enter your Git email address:"
read git_email

# Set the Git username and email globally for all repositories
git config --global user.name "$git_username"
git config --global user.email "$git_email"

# Get the repository name from the current directory (PWD)
repo_name=$(basename "$PWD")

# New method setup access
echo "Enter your GitHub Personal Access Token:"
read access_token

git remote set-url origin https://$access_token@github.com/$git_username/$repo_name




echo "Would you like to enable credential caching? (y/n)"
read enable_cache

# If user chooses 'y', enable Git credential caching for 15 minutes
if [ "$enable_cache" == "y" ]; then
    git config --global credential.helper cache
    echo "Git credentials will be cached for 15 minutes."
else
    echo "Git credentials caching is disabled."
fi




echo "Do you want to configure SSH key authentication for GitHub/GitLab/other Git providers? (y/n)"
read enable_ssh

# If user chooses 'y', proceed to SSH key setup
if [ "$enable_ssh" == "y" ]; then
    # Check if an SSH key already exists
    if [ ! -f "$HOME/.ssh/id_rsa" ]; then
        echo "SSH key not found. Generating a new SSH key..."
        # Generate a new RSA SSH key with the provided Git email address
        ssh-keygen -t rsa -b 4096 -C "$git_email" -f "$HOME/.ssh/id_rsa" -N ""
        echo "SSH key generated successfully."
    else
        echo "SSH key already exists."
    fi

    # Add the SSH key to the SSH agent for authentication
    echo "Adding SSH key to the SSH agent..."
    eval "$(ssh-agent -s)"
    ssh-add "$HOME/.ssh/id_rsa"

    # Display the SSH public key so user can add it to their Git provider account
    echo "Don't forget to add the SSH public key to your Git provider account (GitHub/GitLab/etc.)."
    cat "$HOME/.ssh/id_rsa.pub"
fi




# Display the current Git configuration
echo "Git configuration has been set as follows:"
git config --global --list



echo "Git credentials setup completed successfully!"

