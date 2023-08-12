This documentation is built using Sphinx, and is set up so it can be hosted
on readthedocs.org.

There are a couple of things which enable this:
- The .readthedocs.yaml in the main package directory specifies the build
  environment for the docs to be built in. Please don't edit this file unless
  something in the docs isn't working! It took a lot of trial and error to get
  right, and it is hard to check if it works without pushing to the repository
  and re-running the docs on readthedocs. It does some somewhat weird things,
  such as installing the dependencies for the bolides Python package and
  installing it from source (not PyPI).
- The requirements.txt file in this directory (which is called by
  .readthedocs.yaml) specifies some additional packages which need to be
  installed to get Sphinx to be able to read their classes and documentation
  as well as some packages needed for Sphinx itself to work as configured.

All readthedocs.org does is make the development environment, clone the
package from the specified GitHub repository and branch, and finally build
and host the documentation.

If you are a maintainer with access to the project on readthedocs.org,
you can log into readthedocs.org and on the main page for the project hit
the button to build documentation. This should be done when changes are made
to the package which affect the documentation. You can also set up an
integration with GitHub which will automatically build new docs whenever the
repository updates.
