# Go environment
GOENV = CGO_ENABLED=0 GOOS=linux GOARCH=amd64
GOCMD = ${GOENV} go

# Build info
APP = latency-benchmark
BUILD_NUMBER ?= dev
LDFLAGS = -ldflags "-X main.version=${BUILD_NUMBER}"

# Output directory
DIST_DIR = dist

.PHONY: all build clean test

all: clean build test

build:
	@echo "building ${APP} for linux/amd64..."
	@mkdir -p ${DIST_DIR}
	@${GOCMD} build ${LDFLAGS} -o ${DIST_DIR}/${APP} .
	@echo "build complete: ${DIST_DIR}/${APP}"

clean:
	@echo "cleaning ${APP}..."
	@rm -rf ${DIST_DIR}
	@echo "clean complete"

test:
	@echo "running tests..."
	@go test ./...
	@echo "tests complete"

# Build for current platform (development)
build.local:
	@echo "building ${APP} for local development..."
	@mkdir -p ${DIST_DIR}
	@go build ${LDFLAGS} -o ${DIST_DIR}/${APP} .
	@echo "local build complete: ${DIST_DIR}/${APP}"

# Show build info
info:
	@echo "Target: linux/amd64 (Debian 13 trixie, GLIBC 2.41)"
	@echo "Output: ${DIST_DIR}/${APP}"
	@echo "LDFLAGS: ${LDFLAGS}"
