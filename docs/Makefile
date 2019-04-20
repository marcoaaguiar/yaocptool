# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXAPIDOC  = sphinx-apidoc
SOURCEDIR     = rst
BUILDDIR      = build
CONFIGDIR     = .
PROJECTDIR    = ../yaocptool

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" -c "${CONFIGDIR}" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# In case of clean
clean:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFIGDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@sphinx-apidoc -o "$(SOURCEDIR)" "$(PROJECTDIR)" -f
	@cp $(SOURCEDIR)/modules.rst $(SOURCEDIR)/index.rst
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" -c "$(CONFIGDIR)" $(SPHINXOPTS) $(O)
