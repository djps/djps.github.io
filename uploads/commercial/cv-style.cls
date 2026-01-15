% -- Encoding UTF-8 without BOM

\ProvidesClass{cv-style}[2015/02/27 CV class]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{espanol}{\def\@cv@espanol{}}
\DeclareOption{print}{\def\@cv@print{}}

\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{article}%
}

\ProcessOptions\relax
\LoadClass{article}

%----------------------------------------------------------------------------------------
% Colors %
%----------------------------------------------------------------------------------------

\RequirePackage[usenames,dvipsnames]{xcolor} 

\RequirePackage{calc}
\RequirePackage{fontawesome}

\definecolor{white}{RGB}{255,255,255}

\definecolor{darkgray}{HTML}{333333}
\definecolor{gray}{HTML}{4D4D4D}
\definecolor{lightgray}{HTML}{999999}

\definecolor{blue}{HTML}{66D9EE}
\definecolor{red}{HTML}{FA2772}
\definecolor{orange}{HTML}{FE9720}
\definecolor{green}{HTML}{A7E22E}
\definecolor{purple}{HTML}{9358FE}
\definecolor{aquamarine}{HTML}{36AF90}

\definecolor{date}{HTML}{A4A4A4}

\ifdefined\@cv@print
  \colorlet{blue}{gray}
  \colorlet{red}{gray}
  \colorlet{orange}{gray}
  \colorlet{green}{gray}
  \colorlet{purple}{gray}
  \colorlet{aquamarine}{gray}
  \colorlet{fillheader}{white}
  \colorlet{header}{gray}
\else
  \colorlet{fillheader}{gray}
  \colorlet{header}{white}
\fi
\colorlet{textcolor}{gray}
\colorlet{headercolor}{gray}

%----------------------------------------------------------------------------------------
% Fonts %
%----------------------------------------------------------------------------------------

\RequirePackage{raleway}

\RequirePackage{enumitem}

\RequirePackage[quiet]{fontspec}
% \RequirePackage[math-style=TeX, vargreek-shape=unicode]{unicode-math}
\RequirePackage[math-style=TeX]{unicode-math}

\newfontfamily\bodyfont{Raleway-Regular}[Path=fonts/]
\newfontfamily\bodyfontit{Raleway-LightItalic}[Path=fonts/]
\newfontfamily\thinfont{Raleway-Thin}[Path=fonts/]
\newfontfamily\headingfont{Raleway-Bold}[Path=fonts/]

\defaultfontfeatures{Mapping=tex-text}
\setmainfont[Mapping=tex-text, Color=textcolor, Path = fonts/]{Raleway-Light}

%----------------------------------------------------------------------------------------

\definecolor{date}{HTML}{666666} 
\definecolor{primary}{HTML}{2b2b2b} 
\definecolor{headings}{HTML}{6A6A6A}
\definecolor{subheadings}{HTML}{333333}
\setmainfont[Color=primary, 
Path = fonts/lato/,
BoldItalicFont=Lato-Italic,
BoldFont=Lato-Regular,
ItalicFont=Lato-LightItalic]{Lato-Light}

\setsansfont[Scale=MatchLowercase, Mapping=tex-text, Path = fonts/raleway/]{Raleway-ExtraLight}

% Name command
\newcommand{\namesection}[3]{
\centering{
\fontsize{40pt}{60pt}
\fontspec[Path = fonts/lato/]{Lato-Thin}\selectfont #1 
\fontspec[Path = fonts/lato/]{Lato-Light}\selectfont #2
} \\[5pt]
\centering{
\color{headings}
\fontspec[Path = fonts/raleway/]{Raleway-Regular}\fontsize{11pt}{14pt}\selectfont #3}
\noindent\makebox[\linewidth]{\color{headings}\rule{\paperwidth}{0.4pt}}
\vspace{-15pt}
}

% Section seperators 
\usepackage{titlesec}
\titlespacing{\section}{0pt}{0pt}{0pt} 
\titlespacing{\subsection}{0pt}{0pt}{0pt}
\newcommand{\sectionsep}{\vspace{8pt}}

% Headings command
\titleformat{\section}{\color{headings}
\scshape\fontspec[Path = fonts/lato/]{Lato-Light}\fontsize{16pt}{24pt}\selectfont \raggedright\uppercase}{}{0em}{}

% Subeadings command
\titleformat{\subsection}{
\color{subheadings}\fontspec[Path = fonts/lato/]{Lato-Bold}\fontsize{12pt}{12pt}\selectfont\bfseries\uppercase}{}{0em}{}

\newcommand{\runsubsection}[1]{
\color{subheadings}\fontspec[Path = fonts/lato/]{Lato-Bold}\fontsize{12pt}{12pt}\selectfont\bfseries\uppercase {#1} \normalfont}

% Descriptors command
\newcommand{\descript}[1]{
\color{subheadings}\raggedright\scshape\fontspec[Path = fonts/raleway/]{Raleway-Regular}\fontsize{11pt}{13pt}\selectfont {#1 \\} \normalfont}

% Location command
\newcommand{\location}[1]{
\color{headings}\raggedright\fontspec[Path = fonts/raleway/]{Raleway-Regular}\fontsize{10pt}{12pt}\selectfont {#1\\} \normalfont}
%----------------------------------------------------------------------------------------




\RequirePackage{enumitem}
\RequirePackage{csquotes}

\newcommand{\italica}[1]{%
    {\color{gray}\bodyfontit #1}%
}

%----------------------------------------------------------------------------------------
% Header %
%----------------------------------------------------------------------------------------

\RequirePackage{tikz}

\newcommand{\header}[3]{%
  \begin{tikzpicture}[remember picture, overlay]
    \node [rectangle, fill=fillheader, anchor=north, minimum width=\paperwidth, minimum height=2.2cm] (box) at (current page.north){};
    \node [anchor=center] (name) at (box) {%
      \fontsize{40pt}{65pt}\color{header}%
      {\thinfont #1}{\bodyfont  #2}
    };
    \node[anchor=north, text=pink] (details) at ([yshift=1.5em]box.south) {%
      %\color{header}
      {\small #3}
    };
  \end{tikzpicture}
  \vspace{1cm}
  \vspace{-2\parskip}
}

%\newcommand{\header}[3]{%
%  \begin{tikzpicture}[remember picture,overlay]
%    \node [rectangle, fill=fillheader, anchor=north, minimum width=\paperwidth, minimum height=2.2cm] (box) at (current page.north){};
%    \node [anchor=center] at (box) {%
%      \fontsize{40pt}{65pt}\color{header}%
%      {\thinfont #1}{\bodyfont  #2}
%    };
%    % <-- add this line to put the third argument inside the grey bar
%    
%  \end{tikzpicture}%
%  \vspace{2.2cm}%
%}


%----------------------------------------------------------------------------------------
%	Last updated command %
%----------------------------------------------------------------------------------------

\RequirePackage[absolute, overlay]{textpos}
\RequirePackage{polyglossia}

\RequirePackage[none]{hyphenat}

\RequirePackage{microtype}

\newcommand{\sethyphenation}[3][]{%
  \sbox0{\begin{otherlanguage}[#1]{#2}
    \hyphenation{#3}\end{otherlanguage}}}

\ifdefined\@cv@espanol
  \setdefaultlanguage{spanish}
  \def\lastupdatedtext{Última Actualización el}
\else
  \setdefaultlanguage[variant=british]{english}
  \def\lastupdatedtext{Last Updated on}
\fi

\setlength{\TPHorizModule}{0.01\paperwidth}
\setlength{\TPVertModule}{0.01\paperwidth}

\newcommand{\lastupdated}{ 
  \begin{textblock}{10}(11.3, 0.05)
    \raggedleft
    \fontsize{8pt}{10pt}\color{date}\thinfont 
    \lastupdatedtext{} \today
  \end{textblock}}

%----------------------------------------------------------------------------------------
% Structure %
%----------------------------------------------------------------------------------------
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
        aquamarine\else%
        headercolor\fi%
    } #1#2#3%
  }%
  \stepcounter{colorCounter}%
}

\renewcommand{\section}[1]{
  {\par\vspace{\parskip}
  {%
    \LARGE\headingfont\color{headercolor}%
    \@sectioncolor #1%
  }
  \par\vspace{\parskip}}
}

\renewcommand{\subsection}[2]{
  \par\vspace{.5\parskip}%
  \Large\headingfont\color{headercolor} #2%
  \par\vspace{.25\parskip}%
}

\newcommand{\jobtitle}[1]{%
    {\color{gray}\bodyfontit #1}%
}

\pagestyle{empty}

%----------------------------------------------------------------------------------------
% List environment %
%----------------------------------------------------------------------------------------

\setlength{\tabcolsep}{0pt}
\newenvironment{entrylist}{%
  \begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll}
}{%
  \end{tabular*}
}
\renewcommand{\bfseries}{\headingfont\color{headercolor}}
\newcommand{\entry}[4]{%
  #1&\parbox[t]{12.8cm}{%
    \textbf{#2}%
    \hfill%
    {\footnotesize\addfontfeature{Color=lightgray} #3}\\%
    #4\vspace{\parsep}%
  }\\}

%----------------------------------------------------------------------------------------
% Side block %
%----------------------------------------------------------------------------------------

\setlength{\TPHorizModule}{1cm}
\setlength{\TPVertModule}{1cm}
\newenvironment{aside}{%
  \let\oldsection\section
  \renewcommand{\section}[1]{
    \par\vspace{\baselineskip}{\Large\headingfont\color{headercolor} ##1}
  }
  \begin{textblock}{4.6}(0.25, 1.87)
  \begin{flushright}
  \obeycr
}{%
  \restorecr
  \end{flushright}
  \end{textblock}
  \let\section\oldsection
}

%----------------------------------------------------------------------------------------
% Other tweaks %
%----------------------------------------------------------------------------------------

\RequirePackage[left=5.8cm, top=1.0cm, right=1cm, bottom=0.5cm, nohead, nofoot]{geometry}

    
\setlist[itemize]{align=parleft, left=0em..1em}

\newlength{\bwidth}
\newlength{\mwidth}
\newlength{\pwidth}
\newlength{\diffbwidth}
\newlength{\diffpwidth}

%----------------------------------------------------------------------------------------
% fix hyphenation/justification in itemize
%----------------------------------------------------------------------------------------


\newenvironment{NoHyphItemize}
  {%
    \begin{itemize}[topsep=0pt]
      \hyphenpenalty=10000
      \exhyphenpenalty=10000
      \sloppy
      \setlength\emergencystretch{1em}
  }
  {%
    \end{itemize}
  }
  
  
\usepackage{xcolor,soul,longfbox, framed}  
\definecolor{lightblue}{HTML}{E6F0FA}  % oder ein anderes Hex-/RGB-Value  
\sethlcolor{lightblue}

\newcommand{\myhl}[2][MidnightBlue]{%
  \colorbox{lightblue}{\textbf{\textcolor{#1}{#2}}}%
}

\newcommand{\tighthl}[2][1pt]{%
  {\setlength{\fboxsep}{#1}\hl{#2}}%
}


\newcommand{\newhl}[1]{%
  {\setlength{\fboxsep}{1pt}%
  \colorbox{lightblue}{\textbf{\textcolor{MidnightBlue}{#1}}%
  }%
}
}

%\newcommand{\newhl}[1]{%
%\lfbox[
%  border-width=0.8pt,
%  border-left-color=red,
%  border-style=dotted,
%  padding={0.2ex,0.4ex}, %top&bottom, right&left
%]{%
%    \textbf{\textcolor{MidnightBlue}{#1}}%
%  }%
%}