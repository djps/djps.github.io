\ProvidesClass{academia}[2021/07/03 Academia class]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{print}{\def\@academia@orderbib{}}
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{article}%
}
\ProcessOptions\relax
\LoadClass{article}


%%%%%%%%%%
% Colors %
%%%%%%%%%%

\RequirePackage{xcolor}

\definecolor{white}{RGB}{255,255,255}

\definecolor{darkgray}{HTML}{424242}
\definecolor{gray}{HTML}{424242}
\definecolor{lightgray}{HTML}{878787}

\definecolor{green}{HTML}{C2E15F}
\definecolor{orange}{HTML}{FDA333}
\definecolor{purple}{HTML}{D3A4F9}
\definecolor{red}{HTML}{FB4485}
\definecolor{blue}{HTML}{6CE0F1}
\definecolor{primarycolor}{HTML}{0d3273}
\definecolor{secondarycolor}{HTML}{0c746a}

\colorlet{fillheader}{primarycolor}
\colorlet{header}{white}
\colorlet{textcolor}{gray}
\colorlet{headercolor}{gray}

%%%%%%%%%
% Fonts %
%%%%%%%%%

\RequirePackage[math]{iwona}% math font

\RequirePackage[quiet]{fontspec}
%\RequirePackage[math-style=TeX,vargreek-shape=unicode]{unicode-math}

%\newfontfamily\bodyfont[Ligatures=TeX,Scale=1]{Source Sans Pro}
%\newfontfamily\thinfont[Ligatures=TeX,Scale=1]{Source Sans Pro Light}
%\newfontfamily\headingfont[Ligatures=TeX,Scale=1]{Oswald}
%
%\defaultfontfeatures{Mapping=tex-text}
%\setmainfont[Mapping=tex-text,
%Color=textcolor,Ligatures=TeX,Scale=1]{Source Sans Pro}

%\setmathfont{XITS Math}

%%%%%%%%%%
% Header %
%%%%%%%%%%

\RequirePackage{tikz}

\newcommand{\rolefont}{%
  \fontsize{14pt}{24pt}\selectfont%
  \thinfont%
  \color{white}%
}

\newcommand{\header}[3]{%
  \begin{tikzpicture}[remember picture,overlay]
    \node [rectangle, fill=fillheader, anchor=north, minimum width=\paperwidth, minimum height=2.6cm] (box) at (current page.north){};
    \node [anchor=center] (name) at (box) {%
      \fontsize{24pt}{42pt}\color{header}%
      {\headingfont #1}{\headingfont  #2}
    };
    \node [anchor=north] at (name.south) {%
      \fontsize{10pt}{24pt}\color{header}%
      \thinfont #3%
    };
  \end{tikzpicture}
  \vspace{1.cm}
  \vspace{-2\parskip}
}


%%%%%%%%%%%%%
% Structure %
%%%%%%%%%%%%%
\RequirePackage{parskip}

\newcounter{colorCounter}
\def\@sectioncolor#1#2#3{%
  {%
    \color{%
      \ifcase\value{colorCounter}%
        blue\or%
        red\or%
        orange\or%
        green\or%
        purple\or%
        brown\else%
        headercolor\fi%
    } #1#2#3%
  }%
  \stepcounter{colorCounter}%
}


\pagestyle{empty}


%%%%%%%%%%%%%%%%%%%%
% List environment %
%%%%%%%%%%%%%%%%%%%%

\setlength{\tabcolsep}{0pt}
\newenvironment{entrylist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
% \renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\entry}[4]{%
  \parbox[t]{1.8cm}{#1\hfill}&\parbox[t]{12cm}{%
    \textbf{#2}%
    \hfill%
    {\footnotesize\addfontfeature{Color=lightgray} #3}\\%
    #4\vspace{\parsep}%
  }\\}


%%%%%%%%%%%%%%
% Side block %
%%%%%%%%%%%%%%

\RequirePackage[absolute,overlay]{textpos}
\setlength{\TPHorizModule}{1cm}
\setlength{\TPVertModule}{1cm}
\newenvironment{aside}{%
  \let\oldsection\section
  \renewcommand{\section}[1]{
    \par\vspace{\baselineskip}{\Large\headingfont\color{headercolor} ##1}
  }
  \begin{textblock}{3.8}(1.4, 2.64)
  \begin{flushright}
  \obeycr
}{%
  \restorecr
  \end{flushright}
  \end{textblock}
  \let\section\oldsection
}


%%%%%%%%%%%%%%%%
% Other tweaks %
%%%%%%%%%%%%%%%%


\RequirePackage{geometry}

\RequirePackage{fontawesome5}


%%%%%%%%%%%%%%%%
% Bibliography %
%%%%%%%%%%%%%%%%

\RequirePackage{keyval}


\ifdefined\@academia@orderbib
\RequirePackage[sorting=chronological]{biblatex}%
    \DeclareSortingScheme{chronological}{
      \sort[direction=descending]{\field{year}}
      \sort[direction=descending]{\field{month}}
    }
\else
  \RequirePackage[maxnames=3,sorting=none, backend=biber, giveninits, defernumbers=false,autocite=superscript]{biblatex}%
\fi

\AtEveryBibitem{\clearfield{month}}
\AtEveryCitekey{\clearfield{month}}
\renewcommand{\mkbibnamefamily}[1]{\textsc{#1}}
\renewcommand{\mkbibnamegiven}[1]{\textsc{#1}}
\renewcommand{\mkbibnameprefix}[1]{\textsc{#1}}
\renewcommand{\mkbibnamesuffix}[1]{\textsc{#1}}
\newbibmacro{string+doi}[1]{%
  \iffieldundef{doi}{#1}{\href{http://dx.doi.org/\thefield{doi}}{#1}}}
\DeclareFieldFormat{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}
\DeclareFieldFormat[article]{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}
\DeclareFieldFormat[book]{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}
\DeclareFieldFormat[inproceedings]{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}
\DeclareFieldFormat[misc]{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}
\DeclareFieldFormat[report]{title}{\usebibmacro{string+doi}{{#1}} \addperiod\space}

% \DeclareFieldFormat[article]{title}{#1 \addperiod\space}
% \DeclareFieldFormat[book]{title}{#1 \addperiod\space}
% \DeclareFieldFormat[inproceedings]{title}{#1 \addperiod\space}
% \DeclareFieldFormat[misc]{title}{#1 \addperiod\space}
% \DeclareFieldFormat[report]{title}{#1 \addperiod\space}



\newcommand*{\jourdatedelim}{\addcomma\space}
\newcommand*{\jourserdelim}{\addcomma\space}
\newcommand*{\datevolnumdelim}{\newunitpunct}

\renewbibmacro*{journal+issuetitle}{%
  \usebibmacro{journal}%
  \iffieldundef{series}
    {}
    {\setunit*{\jourserdelim}%
     \printfield{series}%
     \setunit{\jourdatedelim}}%
  \usebibmacro{issue}%
  \usebibmacro{volume+number+eid}%
  \setunit{\addcomma\space}%
  \usebibmacro{date}%
  \newunit}

\renewcommand*{\bibpagespunct}{%
  \ifentrytype{article}
    {\addcolon}
    {\addcomma}%
  \space
}

\DeclareBibliographyDriver{article}{%
    \printnames{author}%
    \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
    \addfontfeature{Color=lightgray}\itshape%
    \usebibmacro{journal+issuetitle}%
    \setunit{\addcomma\space}%%
    \printfield{pages}%
    {\adddot}%
  }
  \par\vspace{0.3\baselineskip}
}


\DeclareBibliographyDriver{book}{%
    \printnames{author}%
    \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
    \addfontfeature{Color=lightgray}\itshape%
    \printlist{publisher}%
    \setunit*{\addcomma\space}%
    \printfield{note}%
    \setunit*{\addcomma\space}%
    \printfield{year}%
    \iffieldundef{location}%
    {}%
    {\setunit{\addcomma\space}%
    \printlist{location}%
    }%
    {\adddot}%
%     \newunit%
  }
  \par\vspace{0.3\baselineskip}
}


\DeclareBibliographyDriver{inproceedings}{%
    \printnames{author}%
    \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
   \addfontfeature{Color=lightgray}%
    \printfield{booktitle}%
    \setunit{\addcomma\space}%
    \printfield{year}%
    \setunit{\addcomma\space}%
    \printlist{location}%
    {\adddot}%
    %\newunit%
  }
  \par\vspace{0.3\baselineskip}
}

\DeclareBibliographyDriver{misc}{%
  \printnames{author}%
    \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
    \addfontfeature{Color=lightgray}\itshape%
    \printfield{booktitle}%
    \setunit*{\addcomma\space}%
    \printfield{note}%
    \setunit*{\addcomma\space}%
    \printfield{year}%
    {\adddot}%
%     \newunit%
  }
  \par\vspace{0.3\baselineskip}
}

\DeclareBibliographyDriver{report}{%
  \printnames{author}%
    \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
   \addfontfeature{Color=lightgray}\itshape%
    \printfield{type}%
    \setunit{\space}%
    \printfield{number}%
    \setunit{\addcomma\space}%
    \printfield{year}%
    {\adddot}%
%     \newunit%
  }
  \par\vspace{0.3\baselineskip}
}


\newcommand{\printbibsection}[2]{
  \begin{refsection}
    \nocite{*}
    \printbibliography[type={#1}, title={\normalfont\headingfont #2}, heading=subbibliography]
  \end{refsection}
}

\newcommand{\printbibsectionkey}[3]{
  \begin{refsection}
    \nocite{*}
    \printbibliography[type={#1}, title={\normalfont\headingfont #2}, keyword={#3}, heading=subbibliography]
  \end{refsection}
}




%%%%%%%%%%%%%%%%%
% Titles %
%%%%%%%%%%%%%%%%%
\RequirePackage{titlesec}

\titleformat{\section}{\Large\headingfont\color{primarycolor}}{}{0.em}{}
\titleformat{\subsection}[runin]{\large\headingfont}{}{0.em}{}[:]
\titleformat{\subsubsection}{\large\headingfont}{}{0.em}{}[:]

\titlespacing{\section}{0pt}{1em}{0.7em}
\titlespacing{\subsection}{0pt}{1em}{0.7em}

%%%%%%%%%%%%%%%%%
% Work packages %
%%%%%%%%%%%%%%%%%

\newcounter{wpcount}[section]

\newcommand{\workpackage}[3]{
\refstepcounter{wpcount}
\vspace*{1em}
\large\bfseries\color{secondarycolor}WP \thewpcount: {#1} (lead: {#2}, co-Is: {#3})
\vspace*{0.3em}
\normalsize\normalfont\color{textcolor}
\newline
}


%%%%%%%%%%%%%%%%%
% Tasks %
%%%%%%%%%%%%%%%%%

\newcounter{taskcount}

% \newenvironment{task}{%
% \refstepcounter{taskcount}
% {\noindent\bfseries{Task \thewpcount.\thetaskcount: }}
% }{\noindent}

\newcommand{\task}[1]{
\refstepcounter{taskcount}
\textbf{Task \thewpcount.\thetaskcount: \normalfont #1}
}


%%%%%%%%%%%%%%%%%
% Wrapping figures %
%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\graphicspath{{fig/}}

\RequirePackage{wrapfig}

\newcommand{\wfig}[4]{
\begin{wrapfigure}{#4}{#3\textwidth}
%   \vspace*{-20pt}
    \includegraphics[width=#3\columnwidth]{#1}
  \caption{#2}
%   \vspace*{-20pt}
\end{wrapfigure}
\par
\hphantom{0.1}\newline
}

\RequirePackage{caption}% math font

\DeclareCaptionLabelSeparator{bar}{ | }
\DeclareCaptionFont{captionlabelfont}{\headingfont}
\DeclareCaptionFont{captionfont}{\bodyfont}
\captionsetup{singlelinecheck=on,justification=justified,  format=plain,
labelfont={color=secondarycolor,captionlabelfont},
font = {captionfont, small},labelsep=bar,
}


\newcommand{\ie}{i.\,e.}
\newcommand{\Ie}{I.\,e.}
\newcommand{\eg}{e.\,g.}
\newcommand{\Eg}{E.\,g.}
\newcommand{\fig}[1]{Fig.~(\ref{#1})}
\newcommand{\equ}[1]{Eq.~(\ref{#1})}