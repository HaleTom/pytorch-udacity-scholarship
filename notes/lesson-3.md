# Lesson 3

![l3-Soumith-Chintala](l3-Soumith-Chintala.png)

### Origins of PyTorch

Soumith Chintala is the creator if PyTorch at Facebook.

His motivation: "How do we have computers understand the world the way we see it?"

Worked on deep learning tooling. Worked on ?EBLearn? at NYU.  Then moved to Torch.  As research moves on, the tools requried also need to progress.

He had a large project (object detection for coco challenge) that was hard to build in Torch, so started work on PyTorch in 2016.

His network was complicated with multiple subsystems and not always differentiable without PyTorch.


Development was Soumith, Adam Paskar and Sam Grosse working under Yann LeCun in FAIR, with was about 75 people.

The design was based on Torch as backen and Chainer (from Japan) front-end.

Slowly access was given to Salesforce, LUA community and Andrej Karpathy. They would give feedback about bugs. This phase was about 6 months.

Word then spread organically. Research labs then started switching.

### Debugging and Designing PyTorch

First public release 1.12.

In about v1.6, Justin Johnson (co-runs CS231 ConvNets course at Stanford) was interning at FAIR.  His networks weren't training because of a non-contiguous tensor passed through a linear layer.

Python is extremely readable and writable, but the trade-off is speed.

The internal autograd engine is written in Python.

A design objective is imperative pythonic code, but at the same time to be as fast as any other framework.

Large parts of PyTorch are written in C++, but everything user-facing is in Python, giving the best of both worlds.

### From research to Production

PyTorch is very popular in the Research community, based upon how PyTorch was built.

The devs welcomed feedback from the research community and was considered very responsive in iterating based on the feedback.

Debugging is very easy. The API is very pythonic, feeling a lot like numpy.

Regarding deployment to production:

If the focus is on production, then the library may not be as usable.

PyTorch 0.4 scaled to thousands of GPUs of parallel training.

1.0 allows exporting the model to a C++ runtime, or quantising (running at 8-bit rather than 32-bit).

For production, function annotations are added to the model. PyTorch will create the model in its own iternal format which can then be shipped to production.

The internal format is an Intermediate Representation which can be run in C++, or in a virtual machine.

After prototyping and testing (takes 90% of the time), the function annotations are added for production deployment.

### Hybrid Frontend

The new PyTorch programming model is called *hybrid frontend* because some parts of the model can be compiled into C++, while others are still in prototyping in Python.

This is powered by a PyTorch's JIT compiler.  Its first aim was to allow easy production-ready export.

The long-term JIT compiler objective is to allow the parts of the model which are compiled to be non-trivially optimised by fusing operations, eg, making memory bandwith bound operations into compute-bound operations.  As new hardware comes available, it can be optimised for larger graphs.

Before 1.0, the ONNX open standard was released. A standard for all deep learning frameworks to talk to each other. Partnered with Microsoft and other big players like Chainer, Caffe2, Tensorflow so that a model trained in one can be exported to another.

Before 1.0, export would be via ONNX to run in somthing like TensorFlow.
But not all complex models could be exported as the standard wasn't developed enough.

### Cutting-edge applications in PyTorch

Andy Brock wrote a paper called Smash where one network would generate the eights that another network would use.  It was like neural network architecture search where all possible architectures and weights needed to be considered.

It took while to work out how to supoprt this.

There's also the FAIR seek project from FaceBook, a text to text transation tool, which supported Heirarchical Story Generation. A story would be given with a very high-level premise, and it would fill in the story with fluent, coherent text.

### User needs and adding features

Researchers ask that when exploring ideas, that there is not a 10x drop in performance.

PyTorch gives an NN LSTM using CUDA NN from Nvidia - about 10x faster than writing an LSTM with for-loops.

NN LSTM doesn't support drop-out, so if they want to experiment with that, there will be a 10x slow-down.

With JIT, the idea is to get the speed of CUDA by stitching in high-performance kernels on the fly in the backend.

Startups and people doing online courses want more interactive tutorials / notebooks with embedded widgets, and first-class integration with colab. Support for Google TPUs. Tensorboard will also have 1st-class integration.

### PyTorch and the Facebook Product

Facebook has FAIR which publishes to arxiv and open source.

Facebook also needs products, eg camera enhancement, machine translation, accessibility interfaces, integrity filtering.

Facebook's view is that these tools may as well be an open-source investment. FAIR's mission is to do AI research in the open and advance humanity using AI, and PyTorch fits with that mission.

### The future of PyTorch

There is already a great community based on PyTorch especially in research and start-ups.

DL is becoming pervasive and essential in many fields. Eg healthcare + data, or computational chemistry, or particle physics.  All are starting to use deep learning, but it's very rudimentary.

Empowering these fields by lowering their barrier of entry is a goal. Providing domain-specific tools.

### Learning more in AI

Advice for people breaking into deep learning / AI:

Be hands on from day one.

He spent a long time trying to collect all the available material, as if having the textbooks or papers would somehow instil knowledge. Passive reading of blog posts with code and beard stroking still produces mental blanks when it comes to writing code.

Prefer hands-on, exploring and producing rather than covering more ground reading or consuming.
