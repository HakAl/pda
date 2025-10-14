#!/bin/bash

# Script to run tests with various options
# Usage: ./run_tests.sh [option]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p tests/logs

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to run tests
run_tests() {
    local test_command=$1
    local description=$2

    print_info "$description"
    echo ""

    if eval "$test_command"; then
        print_success "Tests passed!"
        return 0
    else
        print_error "Tests failed!"
        return 1
    fi
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_error "pytest is not installed. Install with: pip install -r requirements-dev.txt"
    exit 1
fi

# Parse command line argument
case "${1:-all}" in
    all)
        print_info "Running all tests..."
        run_tests "pytest" "All Tests"
        ;;

    unit)
        print_info "Running unit tests only..."
        run_tests "pytest -m unit" "Unit Tests"
        ;;

    integration)
        print_info "Running integration tests only..."
        run_tests "pytest -m integration" "Integration Tests"
        ;;

    fast)
        print_info "Running fast tests (excluding slow tests)..."
        run_tests "pytest -m 'not slow'" "Fast Tests"
        ;;

    slow)
        print_info "Running slow tests only..."
        run_tests "pytest -m slow" "Slow Tests"
        ;;

    coverage)
        print_info "Running tests with coverage report..."
        run_tests "pytest --cov=pda --cov-report=term-missing --cov-report=html" "Tests with Coverage"
        print_info "Coverage report saved to htmlcov/index.html"
        ;;

    coverage-report)
        print_info "Generating coverage report only..."
        pytest --cov=pda --cov-report=html --cov-report=term-missing
        print_success "Coverage report generated at htmlcov/index.html"
        if command -v xdg-open &> /dev/null; then
            xdg-open htmlcov/index.html
        elif command -v open &> /dev/null; then
            open htmlcov/index.html
        fi
        ;;

    verbose)
        print_info "Running tests with verbose output..."
        run_tests "pytest -v" "Verbose Tests"
        ;;

    failed)
        print_info "Re-running last failed tests..."
        run_tests "pytest --lf" "Last Failed Tests"
        ;;

    debug)
        print_info "Running tests with debugging (stops on first failure)..."
        run_tests "pytest -x --pdb" "Debug Mode"
        ;;

    specific)
        if [ -z "$2" ]; then
            print_error "Please specify a test file or pattern"
            echo "Usage: ./run_tests.sh specific tests/test_file.py"
            exit 1
        fi
        print_info "Running specific test: $2"
        run_tests "pytest $2 -v" "Specific Test"
        ;;

    watch)
        print_info "Running tests in watch mode..."
        print_warning "This requires pytest-watch. Install with: pip install pytest-watch"
        if command -v ptw &> /dev/null; then
            ptw -- -v
        else
            print_error "pytest-watch not installed"
            exit 1
        fi
        ;;

    parallel)
        print_info "Running tests in parallel..."
        print_warning "This requires pytest-xdist. Install with: pip install pytest-xdist"
        if python -c "import xdist" 2>/dev/null; then
            run_tests "pytest -n auto" "Parallel Tests"
        else
            print_error "pytest-xdist not installed"
            exit 1
        fi
        ;;

    smoke)
        print_info "Running smoke tests..."
        run_tests "pytest -m smoke" "Smoke Tests"
        ;;

    clean)
        print_info "Cleaning test artifacts..."
        rm -rf .pytest_cache
        rm -rf htmlcov
        rm -rf .coverage
        rm -rf tests/logs/*.log
        rm -rf tests/__pycache__
        rm -rf chroma_db_test
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete
        print_success "Test artifacts cleaned!"
        ;;

    install)
        print_info "Installing test dependencies..."
        pip install -r requirements-dev.txt
        print_success "Test dependencies installed!"
        ;;

    check)
        print_info "Running pre-commit checks..."
        echo ""

        print_info "1. Running linter..."
        if command -v pylint &> /dev/null; then
            pylint pda/ || print_warning "Linting issues found"
        else
            print_warning "pylint not installed, skipping"
        fi

        echo ""
        print_info "2. Running type checker..."
        if command -v mypy &> /dev/null; then
            mypy pda/ || print_warning "Type checking issues found"
        else
            print_warning "mypy not installed, skipping"
        fi

        echo ""
        print_info "3. Running tests..."
        run_tests "pytest -m 'not slow'" "Quick Tests"
        ;;

    help|--help|-h)
        echo "Test Runner for PDA Project"
        echo ""
        echo "Usage: ./run_tests.sh [option]"
        echo ""
        echo "Options:"
        echo "  all              Run all tests (default)"
        echo "  unit             Run unit tests only"
        echo "  integration      Run integration tests only"
        echo "  fast             Run fast tests (exclude slow tests)"
        echo "  slow             Run slow tests only"
        echo "  coverage         Run tests with coverage report"
        echo "  coverage-report  Generate and open coverage report"
        echo "  verbose          Run tests with verbose output"
        echo "  failed           Re-run last failed tests"
        echo "  debug            Run tests in debug mode (stop on first failure)"
        echo "  specific <path>  Run specific test file or pattern"
        echo "  watch            Run tests in watch mode (requires pytest-watch)"
        echo "  parallel         Run tests in parallel (requires pytest-xdist)"
        echo "  smoke            Run smoke tests only"
        echo "  clean            Clean test artifacts and cache"
        echo "  install          Install test dependencies"
        echo "  check            Run pre-commit checks (lint, type check, test)"
        echo "  help             Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh all"
        echo "  ./run_tests.sh coverage"
        echo "  ./run_tests.sh specific tests/test_rag_system.py"
        echo "  ./run_tests.sh specific tests/test_rag_system.py::TestCaching"
        ;;

    *)
        print_error "Unknown option: $1"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo ""
print_info "Done!"