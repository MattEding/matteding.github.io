# Tile Pattern App

Back from my teaching days, I had written some rudimetry code to assist in making
[tile patterns](https://tasks.illustrativemathematics.org/content-standards/tasks/2121)
for my algebra students to make predictions and write equations. Recently I had decided 
to revisit this idea in a more robust manner and it eventually evolved into a Flask app
where you can find a working version deployed on [Heroku](https://tilepattern.herokuapp.com/).

![Pattern Fig. 2](/images/pat1_fig2.png) <!-- make this a gif of the figures -->  
_Color-coding exposes the pattern's structure._

It has definitely been a fun and interesting experience making this come to fruition. 
My plan was to first make a Python script to parse a text file that contains a 
template for the tile pattern. I quickly settled on using "." for a unit tile, 
"-" or "|" for a linear tile, and "O" for a quadratic tile as I felt that these
were easy to type and correspond visually with actual [algebra tiles](https://en.wikipedia.org/wiki/Algebra_tile). For example the pattern above would have a text template of:

```
 .
||O
 .
```

![Algebra Tiles](/images/algtiles.png)  
_Algebra tiles are a nice aide for a wide range of math topics._
_The small squares, rectangles, and large squares correspond to 1, x, and x<sup>2</sup> respectively._

Luckily I had recently researched into [sparse matrices](https://matteding.github.io/2019/04/25/sparse-matrices/) and it didn't take that long 
to realize that block matrices would be the perfect way to take the blueprint 
components. While I could've used `np.block`, I didn't want to figure out what the 
dimensions of the null matrices (blank spaces) and use `np.zeros(shape=?)`. 
Instead `sparse.bmat` abstracts this away by letting me simply use `None` as a 
placeholder for the null matrices, and it would automatically calculate the 
appropriate dimensions. Great! Now I can have the following mappings:

- `"." -> np.array(scalar)`
- `"-" -> np.array(vector)`
- `"|" -> np.array(vector).T`
- `"O" -> np.ones(square_matrix)`

Really cool stuff, but there is still a minor caveat--
I currently have not implemented tiles to align where side lengths would not align
nicely. My plan is to eventually scan rows/columns, find if it needs to be a
unit or variable length based on the "biggest" tile in the given row/column, 
and use slicing on square arrays of 0's to set a restricted subset to 1's. 

Now it was time to make this a viable tool for other teachers to use. I had never 
used `argparse` before and wanted to make a CLI to facilitate usage. While I had 
known about `"__main__"` in Python for quite some time, it was interesting to find out 
about `__main__.py` as a way to make a package into a script! Another point of 
interest was the ability to have `const` and `default` values for a single flag to
assume as its value when the flag was present without a value or when the flag was 
absent respectively. This allowed for nice actions for saving an image to the CWD vs 
an image pop-up while also giving a user the ability to specify a target directory. 

```
>>> python -m tileapp -h
usage: tileapp [-h] [-bw] [-cm COLORMAP] [-o [DIR]] [-p PREFIX] [-v]
               infile dim [dim ...]

Tile pattern parser from txt to png

positional arguments:
  infile                filepath with pattern to parse
  dim                   dimensions to form pattern

optional arguments:
  -h, --help            show this help message and exit
  -a, --alpha           transparency of the colors used in png output; 
                        set to 0 for b/w png
  -cm COLORMAP, --colormap COLORMAP
                        colormap used to differentiate tile parts; see
                        https://matplotlib.org/tutorials/colors/colormaps.html
  -o [DIR], --outdir [DIR]
                        destination file for png output; if omitted, png is
                        popup; if not arg, save png to cwd
  -p PREFIX, --prefix PREFIX
                        prefix used for png output; use alongside outdir
  -v, --verbose         print to stdout the array used for png creation
```

Still not satisfied that there would be much of an audience of math teachers who
would use the then current incarnation as is, I began developing a Flask app 
to make anyone lacking programming knowledge able to reap the benefits of my work. 
For this I had to make a decision regarding how to store the generated images 
to display to the user. Initially I was thinking about taking advantage of 
SQLite's in [in-memory](https://www.sqlite.org/inmemorydb.html) capabilities since
the use case of this application does not need persitent storage. But then again
I thought that while this could work, it seemed like over engineering for a single 
image generated at a time. It was time to reformulate my approach.

<!-- maybe some file themed image here? -->

Most Python programmers are familiar with `open(outfile, 'wb')` and it was time to
put working with files into overdrive. Right off the bat, I decided a `io.BytesIO` would
be the perfect choice since they allow working with image files directly as an object
rather than needed to write to disk. Alas things are never as simple as intially 
thought; HTML was designed to work with plain text. After some sluthing, I stumbled upon 
someone's proposed workaround using a module that I have never looked twice at--`base64`.
In a nutshell, [Base64](https://en.wikipedia.org/wiki/Base64) encodes bytes into text
which makes it ideal to act as a medium between the binary images and my web app.

![Custom Tile Pattern](/images/customtile.png)
_Interface for producing a tile pattern. Need a different figure? Different color?_
_Easy, just change the parameters and you're good to go!_

Despite needing to make the app more exciting with CSS/Bootstrap, those were
secondary to getting my now fully functional app out to the public via Heroku. 
In the past, Heroku had never cooperated with me, and I was unsurprised when it 
invariably threw errors at me. Yet this time I was determined to get it to work 
and _finally_ got it working. My Achilles heel?--thinking that gunicorn was embedded
within Heroku and omitting it from the `requirements.txt`. As the saying goes: "Fool
me once, shame on you; fool me twice, shame on me."

Now all the remains is spicing up the front-end of the app. But that will have to
wait until another day. In the meantime if you are interested in looking at the 
code source visit [my GitHub](https://github.com/MattEding/Tile-Pattern).
