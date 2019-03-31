# Week 1

## Links
* [Udacity Home page](https://classroom.udacity.com/me)
* [Scholarship Home page](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/home)
* [Calendar](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/calendar)
* [GitHub repo](https://github.com/udacity/deep-learning-v2-pytorch)
* [Issue reports (waffle)](https://waffle.io/udacity/dlnd-issue-reports)
* [FAQ](https://udacity.zendesk.com/hc/en-us)
* [AI FAQ](https://udacity.zendesk.com/hc/en-us/sections/360001963971-School-of-Artificial-Intelligence-)
* [Study groups](https://docs.google.com/spreadsheets/d/1D3WarH89WJZu4ASyyZ7DmiaWh_p95QClObUtHXf6JcU/edit#gid=967417831)
* [Slack](https://pytorchscholarsdlnd.slack.com/)

## Career

* [Career page](https://classroom.udacity.com/career/main)

## Contact
* Email contact: deeplearning-support@udacity.com
* If you have reviewed our FAQ and still have a question, you can submit a support request from our help site. https://udacity.zendesk.com/hc/en-us

## Prerequisite knowledge
* [Intro to Data Analysis](Intro to Data Analysis) - Udacity's free numpy pandas matplotlib course

# `conda`

Create new enviroment:

    conda create -n py3 python=3

Activate

    source activate py3

Export environment:

    conda env export > environment.yaml

Create from file:

    conda env create -f environment.yaml

List environments:

    conda env list

Remove environment:

    conda env remove -n env_name

Sharing (in addition to `env export`):

    pip freeze > requirements.txt

Consider making a `py2` environment containing standard packages if using Python 2.x

## Jupyter

Install:

    conda install jupyter notebook

* [nbviewer: Render a notebook](https://nbviewer.jupyter.org/)

Manage conda enviroments inside `jupyter`:
    conda install nb_conda

### Keyboard shortcuts

| Key      | Effect |
|----------|---------|
|C-S-minus   | Split cell at cursor
|L (captal)  | Toggle Line numbers
|C-S-P       | Command Palette (things withoug keyboard shortcuts)
| h          | Keyboard shortcuts


[Magics Documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html)

`%pdb` enables the debugger.

### Slideshows

A very professional-looking way to present a notebook.

View -> Cell Toolbar -> Slideshow

## `numpy`

[Scalars](https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html)

[advanced indexing, boolean indexing, field access, flat iterator](https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html).

[`reshape`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html) tries to return a view of the data. See also `np.newaxis` in indexing above.

`...` expands to as many `:` as neeed to satisfy `ndim`. `None` or `-1` creates a new dimension of size `1`.

### Matrix operations

The results of `dot` and `matmul` are the same *if the matrices are two dimensional*.

`*` a.k.a. `.multiply()` is the element-wise or hadamard product. Use `.dot` for the dot product of 2-D matrices.

`@` a.k.a. `.matmul()` is used if the operands are not 2 dimensional matrices.

[More on the differences of `dot` and `matmul`](https://stackoverflow.com/a/34142617/5353461)


#### Transposes and multiplication

Generally, only use a transpose with multiplication if both original matrices have their data arranged in columns.

Always consider what is represented by the rows and columns in the operands when doing a multiplication.

In numpy, `.T` gives a transpose, but it doesn't rerrange the data in memory. Changing the original matrix will also change the transposed view.
