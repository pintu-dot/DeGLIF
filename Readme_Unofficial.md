-> Codes included are generally compatible with most of the recent versions of pytorch and pytorch geometric.

-> Versions mentioned in `requirements.txt`, are those on which we are sure of code compiling succesfully

-> I have tried running this code on torch 1.10.2 onwards till 2.2.2 so if you have any of these versions already insalled it should mostly run fine. Similary for pytorch geometric it should work fine for versions 2.0.3 to 2.5.3

-> You can check the Cuda version and update the path in "Requirements.ipnyb", accordingly. For example if you have Cuda version 12.1 then in order to install pytorch and related packages, you should change the particular line to `!pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121`

-> Executing Amazon Photo dataset takes most time, and citeseer datasset uses most memory. Execution of cora dataset can be done in most reasonable amount of time (around 2-4 mins depending on device) and can also fit in google colab's provided GPU space.