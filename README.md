# svg2pdfx

This is my attempt to write an SVG-to-PDF parser in under 1k lines of pure Python: a kind of reference implementation that should, hopefully, do a better job compared to most parsers I've seen (for the definition and measure of 'better' please see the benchmarks section below).

At the time when I started writing it, I was beginning to learn SVG and PDF syntaxes. And what better way to learn a syntax than writing a parser for it? Another motivation was my frustration with many existing parsers which didn't quite do what I thought they should.

So, how hard could it be to write such a parser? After all, both SVG and PDF are vector formats, both use Bezier splines for vector graphics, fonts for text and images for, well, images. At least, so I thought when I started. In reality, the job was anything but boring. 

So, if you're interested in the particular task of parsing SVG to PDF, or you are learning SVG, PDF, or both, as I was, or if you just, read on.

# Status

Currently, svg2pdfx supports most of the SVG v1.1 specification (detailed summary will follow soon) and passes the overwhelming majority of the tests in the SVG test suite (see the Benchmarks section below). I am open-sourcing this project at this time in order to try to attract interested developers to help bring this project to shine. The goal is full support of SVG v1.1 except the animated features.

# Benchmarks
