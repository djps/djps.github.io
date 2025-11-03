\ProvidesClass{cv}[2021/07/03 CV class]
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{print}{\def\@academia@orderbib{}}
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{academia}%
    \PassOptionsToClass{orderbib}{academia}%
}
\ProcessOptions\relax
\LoadClass{academia}

\RequirePackage{currfile}
%\edef\mystyle@fontpath{\currfiledir/../commercial/fonts/}

\edef\fontlato{\currfiledir../commercial/fonts/lato/}
\edef\fontraleway{\currfiledir../commercial/fonts/raleway/}

%\renewfontfamily\bodyfont[Ligatures=TeX,Scale=0.9]{Source Sans Pro}
%\renewfontfamily\thinfont[Ligatures=TeX,Scale=0.9]{Source Sans Pro}
%\renewfontfamily\headingfont[Ligatures=TeX,Scale=0.9]{Oswald}
%
%\defaultfontfeatures{Mapping=tex-text}
%\setmainfont[Mapping=tex-text,
%Color=textcolor,Ligatures=TeX,Scale=0.9]{Source Sans Pro}
%\setmonofont[Ligatures=TeX,Scale=0.75]{Anonymous Pro}
%
%\titleformat{\subsection}{\large\bfseries\bfseries}{}{0.7em}{}





%----------------------------------------------------------------------------------------
% Fonts %
%----------------------------------------------------------------------------------------

\RequirePackage{raleway}

\RequirePackage{enumitem}

\RequirePackage[quiet]{fontspec}
% \RequirePackage[math-style=TeX, vargreek-shape=unicode]{unicode-math}
\RequirePackage[math-style=TeX]{unicode-math}

%\makeatletter
%\renewcommand\mystyle@fontraleway{fonts/raleway}
%\renewcommand\mystyle@fontlato{fonts/lato}
%\makeatother

\newfontfamily\bodyfont{Raleway-Regular}[Path=\fontraleway]
\newfontfamily\bodyfontit{Raleway-LightItalic}[Path=\fontraleway]
\newfontfamily\thinfont{Raleway-Thin}[Path=\fontraleway]
\newfontfamily\headingfont{Raleway-Bold}[Path=\fontraleway]

\defaultfontfeatures{Mapping=tex-text}
\setmainfont[Mapping=tex-text, Color=textcolor, Path = \fontraleway]{Raleway-Light}


%----------------------------------------------------------------------------------------


%%%%%%%%%%%%%%%%
% Bibliography %
%%%%%%%%%%%%%%%%


% \RequirePackage{keyval}
% \RequirePackage[natbib=true, maxnames=50, sorting=none, giveninits, defernumbers=true]{biblatex}%defernumbers=true ,
%
% \setlength{\biblabelsep}{-\the\fontdimen2\font}
% \DeclareFieldFormat{labelnumberwidth}{\addspace}

\DeclareFieldFormat[article]{title}{#1 \addperiod\space}
\DeclareFieldFormat[book]{title}{#1 \addperiod\space}
\DeclareFieldFormat[inproceedings]{title}{#1 \addperiod\space}
\DeclareFieldFormat[misc]{title}{#1 \addperiod\space}
\DeclareFieldFormat[report]{title}{#1 \addperiod\space}

% \DeclareBibliographyDriver{article}{%
%   \printfield{title}%
%   \newblock%
%   \printnames{author}%
%   \par%
%   \newblock%
%   {%
%     \footnotesize\addfontfeature{Color=lightgray}\itshape%
%     \usebibmacro{journal+issuetitle}%
%     \setunit{\space}%
%     \printfield{pages}%
%     \newunit%
%     \printlist{publisher}%
%     \setunit*{\addcomma\space}%
%     %\printfield{year}%
%     \newunit%
%   }
%   \par\vspace{0.3\baselineskip}
% }


\DeclareBibliographyDriver{article}{%
  \printnames{author}%
  \addperiod\space
  \newblock%
  \itshape\printfield{title}%
  \newblock%
  {%
    \addfontfeature{Color=lightgray}\itshape%
    \usebibmacro{journal+issuetitle}%
    \setunit{\space}%
    \printfield{pages}%
    %     \newunit%
    %     \printlist{publisher}%
    %\setunit*{\addcomma\space}%
    %\printfield{year}%
    %\newunit%
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
    \setunit{\addcomma\space}%
    \printlist{location}%
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
    %     \newunit%
  }
  \par\vspace{0.3\baselineskip}
}

\DeclareBibliographyDriver{software}{%
  \printnames{author}%
  \setunit*{\addperiod\space}%
  \newblock%
  \printfield{title}%
  \newblock%
  {%
    \addfontfeature{Color=lightgray}%
    \setunit*{\addcomma\space}%
    \texttt{\printfield{note}}%
    \setunit*{\addcomma\space}%
    \printfield{year}%
    %     \newunit%
  }
  \par\vspace{0.3\baselineskip}
}

% \DeclareNameFormat{author}{%
%   \addfontfeature{Color=lightgray}%
%   \ifthenelse{\value{listcount}<\value{liststop}\namepartgiven\space\namepartfamily\addcomma\space}
% }


% \DeclareNameFormat{author}{%
% \addfontfeature{Color=lightgray}%
% %     \nameparts{#1}
%      \usebibmacro{name:family}
%       {\namepartfamily}
%       {\namepartgiven}
%       {\namepartprefix}
%       {\namepartsuffix}%
%      }


\DeclareSortingScheme{chronological}{
  \sort[direction=descending]{\field{year}}
  \sort[direction=descending]{\field{month}}
}


% \newcommand{\printbibsection}[2]{
%   \begin{refsection}
%     \nocite{*}
%     \printbibliography[type={#1}, title={#2}, heading=subbibliography]
%   \end{refsection}
% }

