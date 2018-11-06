.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt
	pip install -e .

.PHONY: check-node
check-node:
	node --version

.PHONY: fetch-ref
fetch-ref: check-node
	python tests/get_reference.py

.PHONY: test
test: fetch-ref
	pytest

.PHONY: fmt
fmt:
	black arbor tests

.PHONY: lint
lint:
	flake8 arbor tests
	black --check arbor tests
