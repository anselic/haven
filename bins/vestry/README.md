# vestry

vestry is a build system and project manager for haven projects.

## Dependency

You can install dependencies by:

```toml
[project]
name = "..."
# ...

[dependencies]
package-name = { version = "1.0.0", path = "path/to/package" }
```

## Build scripts

A vestry project can run a Haven script before compilation and a separate packaging script after its artifact exists:

```toml
[project]
# ...
build = "build.hv"
post-build = "package.hv"
```

`build.hv` can contribute native linker configuration by printing directives:

```hv
proc main() i32 {
    println("vestry::link-search=native=path/to/lib");
    println("vestry::link-lib=SDL3");
    println("vestry::link-archive=path/to/libhelper.a");
    println("vestry::link-arg=-pthread");
    println("vestry::warning=building bundled native dependency");
    return 0;
}
```

Relative search and archive paths resolve from the package root. Native link
directives emitted by libraries travel through their `.hvmeta` artifacts to the
final executable.
Both hooks receive `VESTRY_PROJECT_ROOT`, `VESTRY_TARGET_DIR`,
`VESTRY_TARGET_OS`, `VESTRY_TARGET_ARCH`, `VESTRY_OUTPUT_KIND`,
`VESTRY_PKG_NAME`, and `VESTRY_PKG_VERSION`.

`build` additionally receives `VESTRY_OUT_DIR`

`post-build` receives `VESTRY_ARTIFACT` and cannot change
linking.