# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Bash
#     language: bash
#     name: bash
# ---

# hide
# skip
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_slow
# -

# # Pull requests made easy
# > Making your first pull request to fastai
#
# - hide_colab_badge: true
# - image: images/pr-head.png

# hide
exit  # don't run this by accident!

# In order to contribute to fastai (or any fast.ai library... or indeed most open source software!) you'll need to make a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests), also known as a *PR*. Here's an [example](https://github.com/fastai/fastai/pull/2648) of a pull request. In this case, you can see from the description that it's something that fixes some typos in the library. If you click on "Files changed" on that page, you can see all the changes made. We get notified when a pull request arrives, and after checking whether the changes look OK, we "merge" it (which means that we click a button in GitHub that causes all those changes to get automatically added to the repo).
#
# Making a pull request for the first time can feel a bit over-whelming, so I've put together this guide to help you get started. We're going to use GitHub's command line tool `gh`, which makes things faster and easier than doing things through the web-site (in my opinion, at least!) The easiest way to install `gh` on Linux (works on any distribution, including Ubuntu) is using `conda`:

conda install - yc fastai gh

# I'm assuming in this guide that you're using Linux, and that you've already got [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) set up. This is the default in all of the fast.ai course guides, and is highly recommended. It should work fine on Mac too, although I haven't tested it. On Windows, use [Ubuntu on WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10). To install `gh` on Mac type: `brew install github/gh/gh`.
#
# > This document is created from a Jupyter Notebook. You can run the notebook yourself by getting it [from here](https://github.com/fastai/fastai/blob/master/nbs/dev-setup.ipynb). You'll need to install the [Jupyter Bash kernel](https://github.com/takluyver/bash_kernel).

# ## One time only setup

# ### Setting up access and `gh`

# To develop fastai, you'll need to install `fastai` and `nbdev` (this also checks you have the latest versions, if they're already installed):

conda install - y - c fastai - c pytorch - c anaconda anaconda fastai nbdev

# **NB**: if you're using miniconda instead of Anaconda, remove `-c anaconda anaconda` from the above command.

# [nbdev](https://nbdev.fast.ai) is a framework for true literate programming; see [this article](https://www.fast.ai/2019/12/02/nbdev/) to learn why you should consider using it for your next project (and why we use it for fastai).
#
# You'll need to set up `ssh` access to GitHub, if you haven't already. To do so, follow [these steps](https://docs.github.com/en/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account). Once you've created an ssh key (generally by running `ssh-keygen`), you can copy the contents of your `.~/ssh/id_rsa.pub` file and paste them into GitHub by clicking "New SSH Key" on [this page](https://github.com/settings/keys).
#
# Once that's working, we need to get a [personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) to allow `gh` to connect to GitHub. To do so, [click here](https://github.com/settings/tokens/new) and enter "gh" in the "Note" section, and click the `repo`, `read:discussion`, and `read:org` checkboxes (note that `gh` can do this automatically for you, but it only works conveniently if you're running the code on your local machine; most fastai developers are probably using a remote GPU server, such as Paperspace, AWS, or GCP, so we show the approach below because it works for everyone).

# <img alt="Personal access token screen" width="495" caption="Personal access token screen" src="images/att_00000.png">

# Then click "Generate Token" at the bottom of the screen, and copy the token (the long string of letters and numbers shown). You can easily do that by clicking the little clipboard icon next to the token.

# <img alt="Copying your token" width="743" caption="Copying your token" src="images/att_00001.png">

# Now run this in your shell, replacing `jph01` with your GitHub username, and the string after `TOKEN=` with your copied token:

GH_USER = jph01
TOKEN = abae9e225efcf319f41c68f3f4d7c2d92f59403e

# Setup `gh` to use `ssh` to connect to GitHub, so you don't have to enter you name and password all the time:

gh config set git_protocol ssh

# Create your GitHub authentication file:

echo - e "github.com:\n  user: $GH_USER\n  oauth_token: $TOKEN\n" > ~ / .config / gh / hosts.yml

# ### Set up `fastcore`

# Now we're ready to clone the `fastcore` and `fastai` libraries. We recommend cloning both, since you might need to make changes in, or debug, the `fastcore` library that `fastai` uses heavily. First, we'll do `fastcore`:

gh repo clone fastai / fastcore

# We update our installed version to use [editable mode](https://stackoverflow.com/questions/35064426/when-would-the-e-editable-option-be-useful-with-pip-install). This means that any changes you make to the code in your checked-out repo will automatically be used everywhere on your computer that uses this library:

cd fastcore
pip install - qe .

# We update the repo to create and use a [fork](https://medium.com/@tharis63/git-fork-vs-git-clone-8aad0c0e38c0#:~:text=Git%20Fork%20means%20you%20just,to%20your%20own%20GitHub%20profile.&text=Then%20make%20your%20changes%20and,it%20to%20the%20main%20repository.):

gh repo fork - -remote

# Because all fast.ai libraries use nbdev, we need to run [nbdev_install_git_hooks](https://nbdev.fast.ai/cli#Git-hooks) the first time after we clone a repo; this ensures that our notebooks are automatically cleaned and trusted whenever we push to GitHub:

nbdev_install_git_hooks

# ### Set up `fastai`

# Now we'll do the same steps for `fastai`:

cd ..
gh repo clone fastai / fastai
cd fastai

# We'll do an editable install of `fastai` too:

pip install - qe .[dev]

# ...and fork it and install the git hooks:

gh repo fork - -remote

nbdev_install_git_hooks

# ## Creating your PR

# Everything above needs to be done just once. From here on are the commands to actually create your PR.
#
# Create a new [git branch](https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging), by running the following, replacing `test-pr` with the name you want to give your pull request (use something that will be easy for you to remember in the future if you need to update your PR):

git checkout - b test - pr

# Make whatever changes you want to make in the notebooks, and remember to run `nbdev_build_lib` when you're done to ensure that the libraries are built from your notebook changes (unless you only changed markdown, in which case that's not needed). It's also a good idea to check the output of `git diff` to ensure that you haven't accidentally made more changes than you planned.
#
# When you're ready, `commit` your work, replacing "just testing" here with a clear description of what you did in your commit:

git commit - am "just testing"

# The first time you push from your fork, you need to add `-u origin HEAD`, but after the first time, you can just use `git push`.

git push - u origin HEAD

# Now you're ready to create your PR. To use the information from your commit message as the PR title, just run:

gh pr create - f

# To be interactively prompted for more information (including opening your editor to let you fill in a detailed description), just run `gh pr create` without the `-f` flag. As you see above, after it's done, it prints the URL of your new PR - congratulations, and thank you for your contribution!

# <img alt="The completed pull request" width="615" caption="The completed pull request" src="images/att_00002.png">

# ## Post-PR steps

# To keep your fork up to date with the changes to the main fastai repo, and to change from your `test-pr` branch back to master, run:

git pull upstream master
git checkout master

# In the future, once your PR has been merged or rejected, you can delete your branch if you don't need it any more:

git branch - d test - pr
