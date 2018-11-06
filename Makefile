.PHONY: install
install: clean
	pip install .

.PHONY: install-dev
install-dev: clean
	pip install -r requirements.txt
	pip install -e .

.PHONY: check-node
check-node:
	node --version

.PHONY: fetch-ref
fetch-ref: check-node
	cd tests && python get_reference.py

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

.PHONY: clean
clean:
	rm -rf .eggs
	rm -rf .pytest_cache
	rm -rf *.egg-info
