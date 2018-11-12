.PHONY: install
install: clean
	pip install .

.PHONY: install-dev
install-dev: clean
	pip install -r requirements.txt
	pip install -e .

.PHONY: node-installed
node-installed:
	cd tests/arbor-harness && node --version

.PHONY: harness
harness:
	[ -d tests/arbor-harness/data ] || git submodules --init --recursive

.PHONY: harness-install
harness-install: harness node-installed
	cd tests/arbor-harness && [ -d node_modules ] || npm install

.PHONY: harness-populate
harness-populate: harness-install
	cd tests/arbor-harness && [ -d data/3034133/results ] || npm start

.PHONY: test
test: harness-populate
	pytest -v

.PHONY: test-quick
test-quick: harness-populate
	pytest -v --skipslow

.PHONY: fmt
fmt:
	black arbor tests setup.py

.PHONY: lint
lint:
	flake8 arbor tests
	black --check arbor tests setup.py

.PHONY: clean
clean:
	rm -rf .eggs
	rm -rf .pytest_cache
	rm -rf *.egg-info
