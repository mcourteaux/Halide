include ../support/Makefile.inc

GENERATOR_OUTPUTS=stmt_html,conceptual_stmt_html,static_library,c_header
.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/perlin

$(GENERATOR_BIN)/perlin.generator: generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/perlin.a: $(GENERATOR_BIN)/perlin.generator
	@mkdir -p $(@D)
	$^ -g perlin -e $(GENERATOR_OUTPUTS) -o $(@D) -f perlin target=$*-cuda-profile

$(BIN)/%/perlin: perlin.cpp $(BIN)/%/perlin.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) $(IMAGE_IO_FLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS)

$(BIN)/%/out.png: $(BIN)/%/perlin
	@mkdir -p $(@D)
	$< $@

clean:
	rm -rf $(BIN)

test: $(BIN)/$(HL_TARGET)/out.png
