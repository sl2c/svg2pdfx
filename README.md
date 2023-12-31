# svg2pdfx

This is my attempt to write an SVG-to-PDF parser in under 1k lines of pure Python: a kind of reference implementation that should, hopefully, do a better job compared to most parsers I've seen (for the definition and measure of 'better' please see the benchmarks section below).

At the time when I started writing it, I was beginning to learn SVG and PDF syntaxes. And what better way to learn a syntax than writing a parser for it? Another motivation was my frustration with many existing parsers which didn't quite do what I thought they should.

So, how hard could it be to write such a parser? After all, both SVG and PDF are vector formats, both use Bezier splines for vector graphics, fonts for text and images for, well, images. At least, so I thought when I started. In reality, the job was anything but boring. 

So, if you're interested in the particular task of parsing SVG to PDF, or you are learning SVG, PDF, or both, as I was, read on. I will try to make it interesting.

# Usage

The module can be used as a class library, or as a command-line tool:

```bash
svg2pdfx.py [-debug] [-o output.pdf] input1.svg [input2.svg ...]
```

This invocation converts several SVG files into a single PDF file.

# Features

## Use of XObject-s

What immediately sets svg2pdfx apart from other such converters is the fact that if the same (see below for the definition of 'same') object is defined multiple times in the same SVG file or multiple SVG files on the command line, they will be defined just once in output PDF. This is done using XObject-s. This may help significantly reduce the resulting file size in some cases.

A note on Adobe Acrobat/Reader: while Adobe's products seem to use caching for rendering vector fonts, just as their own PDF Reference (ver.1.7, sec. 5.1) suggests:

> Programmers who have experience with scan conversion of general shapes may be concerned about the amount of computation that this description seems to imply. However, this is only the abstract behavior of glyph descriptions and font programs, not how they are implemented. In fact, an efficient implementation can be achieved through careful caching and reuse of previously rendered glyphs.

no caching is done for rendering XObject-s. So, if there are lots of them rendering may be slow. However, you can use other tools (like e.g. ghostscript) to easily convert (dereference) XObject-s into page contents stream operators.

# Status

Currently, svg2pdfx supports most of the SVG v1.1 specification (detailed summary will follow soon) and passes the overwhelming majority of the tests in the SVG test suite (see the Benchmarks section below). I am open-sourcing this project at this time in order to try to attract interested developers to help bring this project to shine. The goal is full support of SVG v1.1 except the animated features.

# Benchmarks
