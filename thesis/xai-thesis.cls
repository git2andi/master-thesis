% =============================================================================
% Documentclass for theses at the chair of Explainable Machine Learning at the University of Bamberg
% 
% Version History:
% 2016-09-28        Holger Graf, based on diplarb.sty by Christoph Gollmick (05.09.2002) 
% 2020-10-27  	    Holger Graf
%                   Licence CC BY 4.0
%					Link: https://www.overleaf.com/latex/templates/thesis-template-microeconomics-at-fsu-jena/shqhkgcqtvsn
%					Accessed 18.01.2023, original name of this file: micro-econ-thesis.cls
% 2023-02-01		Ines Rieger, adapted and renamed file
%                   Licence CC BY 4.0
%					Lehrstuhl für Erklärbares Maschinelles Lernen
%					Universität Bamberg
%
%
% =============================================================================
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{xai-thesis}[2023/02/01 v1 xai-thesis]
\typeout{Document class for theses at the chair of Explainable Machine Learning at the University of Bamberg}
\LoadClass[12pt,a4paper]{article}
\usepackage{ifthen}
\usepackage{tabularx}
\usepackage[svgnames]{xcolor}


% ----------------------------------------------------------------------------
% Seitenraender, Abstaende, Zaehler
% ----------------------------------------------------------------------------
\setlength{\topmargin}{0.0cm}
\setlength{\textheight}{23.0cm}
\setlength{\textwidth}{15.0cm}
\setlength{\headsep}{0.8cm}
\setlength{\oddsidemargin}{0.9cm}
\addtolength{\evensidemargin}{-2.49cm}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 2pt minus 1pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}
% Damit mehr Tabellen/Abbildungen auf eine Seite passen. (Kopka-Buch S. 170)
\setcounter{topnumber}{9}
\setcounter{totalnumber}{9}
% grosse Objekte nicht erst auf letzter Seite... (siehe Kopka-Buch S. 170)
\renewcommand{\topfraction}{0.99}
\renewcommand{\textfraction}{0.01}
% typische Abkuerzungen
\newcommand{\zB}{z.\,B.\ }
\newcommand{\idR}{i.\,d.\,R.\ }
\newcommand{\bzw}{bzw.\ }
\newcommand{\etc}{etc.\ }
\newcommand{\iA}{i.\,A.\ }
\newcommand{\uU}{u.\,U.\ }
\newcommand{\ua}{u.\,a.\ }
\newcommand{\usw}{usw.\ }
\renewcommand{\dh}{d.\,h.\ }
\newcommand{\ie}{i.\,e.\ }
\newcommand{\eg}{e.\,g.\ }
% Hurenkinder und Schusterjungen verbieten
\clubpenalty = 10000
\widowpenalty = 10000
\displaywidowpenalty = 10000

% ----------------------------------------------------------------------------
% Schnittstellen-Parameter
% ----------------------------------------------------------------------------
% interne Makros initialisieren/vorbelegen
\newcommand{\@thesisTitle}{}
\newcommand{\@thesisType}{}
\newcommand{\@thesisAuthor}{}
\newcommand{\@thesisMail}{}
\newcommand{\@thesisGrade}{}
\newcommand{\@thesisFirstSupervisor}{}
\newcommand{\@thesisSecondSupervisor}{}
\newcommand{\@thesisDate}{}
\newcommand{\@thesisMatrikel}{}

% aktuelle Daten uebernehmen
\newcommand{\thesisTitle}[1]{\renewcommand{\@thesisTitle}{#1}}
\newcommand{\thesisType}[1]{\renewcommand{\@thesisType}{#1}}
\newcommand{\thesisAuthor}[1]{\renewcommand{\@thesisAuthor}{#1}}
\newcommand{\thesisGrade}[1]{\renewcommand{\@thesisGrade}{#1}}
\newcommand{\thesisFirstSupervisor}[1]{\renewcommand{\@thesisFirstSupervisor}{#1}}
\newcommand{\thesisSecondSupervisor}[1]{\renewcommand{\@thesisSecondSupervisor}{#1}}
\newcommand{\thesisDate}[1]{\renewcommand{\@thesisDate}{#1}}

% ----------------------------------------------------------------------------
% Erzeugung des Deckblatts
% ----------------------------------------------------------------------------
\newcommand{\thesisMakeTitle}[0]{
  \cleardoublepage
  \pagenumbering{alph}
  \pagestyle{empty}
  \null
  \begin{minipage}{6cm}
  	\includegraphics[width=5.8cm]{xaiLogo.png}
  \end{minipage}
  \hfill
  \begin{minipage}{4cm}
	\includegraphics[width=4cm]{Otto-Friedrich-University_Bamberg_logo.png}
  \end{minipage}
  \begin{center}
    \Large
    {\huge\textbf{\@thesisTitle}\par}
    \vfill
      \textbf{\@thesisType{}} \\
      \vfill
      \@thesisGrade
      \vfill
      \@thesisAuthor \\[1em]
	  \vfill
      {\large \@thesisDate} \\
      \vfill

	\begin{minipage}[t]{\textwidth} 
    {\normalsize
      
     \textbf{Supervisor:}\\[1ex]        
      \ifthenelse{\equal{\@thesisFirstSupervisor}{}}{}{1st: \@thesisFirstSupervisor \\}
      \ifthenelse{\equal{\@thesisSecondSupervisor}{}}{}{2nd: \@thesisSecondSupervisor \\}
      
	  \smallskip
	  Chair of Explainable Machine Learning \\
      Faculty of Information Systems and Applied Computer Sciences \\
      Otto-Friedrich-University Bamberg \\
      \par
    }
	\end{minipage}
  \end{center}
  % Setup des hyperref-Pakets vervollst"andigen (Autor, Titel)
  \hypersetup{
    pdfauthor = {\@thesisAuthor}, % Author
    pdftitle = {\@thesisTitle}, % Titel
  }
}

% ----------------------------------------------------------------------------
% Erzeugung der Selbststaendigkeits-Erklaerung
% ----------------------------------------------------------------------------
\newcommand{\makeThesisDeclaration}{
  \cleardoublepage
  \thispagestyle{empty}
  \subsection*{Declaration of Authorship}
 Ich erkl\"are hiermit gem\"a\ss~\S9 Abs. 12 APO, dass ich die vorstehende Abschlussarbeit 	 selbstst\"andig verfasst und keine anderen als die angegebenen Quellen und Hilfsmittel benutzt habe. Des Weiteren erkl\"are ich, dass die digitale Fassung der gedruckten Ausfertigung der Abschlussarbeit ausnahmslos in Inhalt und Wortlaut entspricht und zur Kenntnis genommen wurde, dass diese digitale Fassung einer durch Software unterst\"utzten, anonymisierten Pr\"ufung auf Plagiate unterzogen werden kann.\\[2ex]

\begin{tabularx}{\textwidth}{@{}p{4cm}Xp{4cm}@{}}
	& &   \\ 
\cline{1-1}\cline{3-3}
Place, Date  & & Signature  \\
\end{tabularx}
} 