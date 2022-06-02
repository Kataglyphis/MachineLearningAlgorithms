# LSTM

<h1 align="center">
  <br>
  <a href="https://jotrocken.blog/"><img src="images/logo.png" alt="OpenGLEngine" width="200"></a>
  <br>
  Basic LSTM implementation
  <br>
</h1>

<h4 align="center">Build a sequence of characters <a href="https://jotrocken.blog/" target="_blank"></a>.</h4>

<p align="center">
  <a href="https://paypal.me/JonasHeinle?locale.x=de_DE">
    <img src="https://img.shields.io/badge/$-donate-ff69b4.svg?maxAge=2592000&amp;style=flat">
  </a>
</p>

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#download">Download</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
      <ul>
        <li><a href="#key-features">Key Features</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

From Kants critique of practical reason I got sentences like:

(emb_size = 16
hidden_size = 124
seq_length = 512
learning_rate = 5e-2 
max_updates = 500000
batch_size = 32)

1.Result:
"on this it thush that determinated and representakoug advantaresuct
form) usefing
on which there dayver in this made of us as all in our cases of a rat just on it were orle freptive the law is of it. Thus to formalore this serucion the vatures,
the conding our explesed the vater, he constadnes and only a necessare the hedution now thisk the conduapo-sted anduredsifinated are
lastingavt one of thisplyssul supposables outisaty his
oblige of us. Forble only a have feeling to it it itself vione in it were is can in it were in this
doublious merew, justifysero anderedores in our vabutualy as a new this
condualed as alwagation as a
categ- will gerery give
us
all the fationed as all caneon constanally a universal empithed for such it it be advantaking on it whrows for
not of in this
sured, or beings have or private it it it tho advinated in a noursinedmegules determinour now on it. Thus, fromnevery or in a prove of thishity or if and it of a raterfined to formare of this deterd no douneablentonedered it come is a higher deterd of this dul to high. Now
its conduaventi;ethul frequently screp; this
suterninamersthicatedmentare the om its oblinately is a higherw
somethicated as alwaysion firmtom constionchilesogich is in this
dodevere
composenery
thind the bas ratherw
hablog constioning virtuest trued it out to
formapo,
advant well of and chereng in its or if thishs in our causaly his
erred oursualing the
ablegt of would only a universul the formais or segve
frounds thishigances the vatherw
the bat the determination of to its or in it because the caterfinateorangionation with things the deduct
of morality this dod alonestimateneables, notions our ear
of it a had of us a had only only a law with the
bein the in
happly to finds to
acu thereon whiculain he can physist to this dut all the vated, however, of it it destricest, and man's onowledgringuit is nothing
an every us to fog mo,ing on its own adept only a reasonally andering the
tysted a universublestlying the bound fromnes"  

2.Result:
"on this it thush that determinated and representakoug advantaresuct
form) usefing
on which there dayver in this made of us as all in our cases of a rat just on it were orle freptive the law is of it. Thus to formalore this serucion the vatures,
the conding our explesed the vater, he constadnes and only a necessare the hedution now thisk the conduapo-sted anduredsifinated are
lastingavt one of thisplyssul supposables outisaty his
oblige of us. Forble only a have feeling to it it itself vione in it were is can in it were in this
doublious merew, justifysero anderedores in our vabutualy as a new this
condualed as alwagation as a
categ- will gerery give
us
all the fationed as all caneon constanally a universal empithed for such it it be advantaking on it whrows for
not of in this
sured, or beings have or private it it it tho advinated in a noursinedmegules determinour now on it. Thus, fromnevery or in a prove of thishity or if and it of a raterfined to formare of this deterd no douneablentonedered it come is a higher deterd of this dul to high. Now
its conduaventi;ethul frequently screp; this
suterninamersthicatedmentare the om its oblinately is a higherw
somethicated as alwaysion firmtom constionchilesogich is in this
dodevere
composenery
thind the bas ratherw
hablog constioning virtuest trued it out to
formapo,
advant well of and chereng in its or if thishs in our causaly his
erred oursualing the
ablegt of would only a universul the formais or segve
frounds thishigances the vatherw
the bat the determination of to its or in it because the caterfinateorangionation with things the deduct
of morality this dod alonestimateneables, notions our ear
of it a had of us a had only only a law with the
bein the in
happly to finds to
acu thereon whiculain he can physist to this dut all the vated, however, of it it destricest, and man's onowledgringuit is nothing
an every us to fog mo,ing on its own adept only a reasonally andering the
tysted a universublestlying the bound fromnes"

3.Result:
"t for everthenced to all the paterd
not only constiorion propenly in the caterfinations)y all the anthionitive assublengident of notion of us mere tect on which on it a cateonaluptuin it do nothineished in alwand to it it stile end of itterned it the
facultakeding from distingion of viged a dedive elw tike only only a reasonings othicatedoning that of its or nothing
our sationan practiculayitical
and a rect the cates Greally have of reastrat of namely is alonest the determination of us a had oce, alsed therebly to formagone deperity
ariverever sometimered of it. Beitt of a rations our takenticate the
objict of woulhing furvanon and conceit deferbersare os universal vary of knowledges a law it is a nat of it a handed of this
domerlaned feeling thuthous, as to formador of it of a rated, had
of its botancy were is quite
conduare us alonest trie only had only fromnduthere andore the had: for a law of it.

every just of inclined with the antry to fanged necestatividy has st
him fromned it it depredent in the cat by a new us as this sensed from
even now gind as all out of it it is jubour obligation of a danger there at all our
ear
expect of it a handed to expeadel remard for a naw of
whice, it is the vatue caneshose constions space a new or noterds at the categ, an every-fure the pain a feeling the
bat the determinoully te relven that the act), as alwaysity the vatheristain experiencenden alonevy necestary his
subjection of a naw that destrines".
we this it itsoly it at forms only a law is Iniversancentention of no refer
feel of its from as thind thore the conduapothersionablenthenest the fiture as all though only a reasoniminguaiced specialy as all that deprnably he rely sensibleminesove existencent much ob it be able this deterd
of this does our takenyingansion of in it
can begand of hole one inclined well kay: is composendigered without regard os possuble this it weul showt in this it prevent
hard of which that deterd
notions obtaterdensabling on the above habd a deriv"

<!-- [![Kataglyphis Engine][product-screenshot]](https://jotrocken.blog/)-->

Implementing a basic LSTM net for generating char sequences.

### Key Features


|          Feature                |   Implement Status | ◾ Other Configs |
| --------------------------------| :----------------: | :-------------: |
| Backpropagation                 |         ✔️         |        ❌      |

### Built With

* [Python](https://www.python.org/)
* [CuPy](https://cupy.dev/)
* [Cuda](https://developer.nvidia.com/cuda-zone)


<!-- GETTING STARTED -->
## Getting Started

You might only clone the repo and get to go immediately :)

### Prerequisites

* [Python](https://www.python.org/)
* [CuPy](https://cupy.dev/)
* [Cuda](https://developer.nvidia.com/cuda-zone)

### Installation

1. Clone the repo
   ```sh
   git clone git@github.com:Kataglyphis/LSTM.git
   ```


<!-- USAGE EXAMPLES -->
## Usage

_For more examples, please refer to the [Documentation](https://jotrocken.blog/)_

Run "python lstm_cupy.py train" for training

Run "python lstm_cupy.py gradcheck" for checking the gradients

Run "python lstm_cupy.py sample" for generating sample

Under folder "data/" one can place any kind of .txt file one want to train the net with.
Just make sure it is a .txt!


<!-- ROADMAP -->
## Roadmap
Upcoming :)
<!-- See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues). -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the GPL-3.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Jonas Heinle - [@your_twitter](https://twitter.com/Cataglyphis_) - jonasheinle@googlemail.com

Project Link: [https://github.com/Kataglyphis/OpenGLEngine](https://github.com/Kataglyphis/OpenGLEngine)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
This work was inspired by previous work on char-rnn. Especially to mention:
* [karpathy](https://github.com/karpathy/char-rnn) <br>
* [quanpn90](https://github.com/quanpn90/LSTMAssignment-DLNN2020)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/jonas-heinle-0b2a301a0/
[product-screenshot]: images/Screenshot.png