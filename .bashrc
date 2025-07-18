#!/bin/bash
SHELL=/bin/bash
shopt -s expand_aliases  # for trying to get cron to run aliases

# ~/.bashrc: executed by bash(1) for non-login shells.
# see /usr/share/doc/bash/examples/startup-files (in the package bash-doc)
# for examples

# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

# don't put duplicate lines or lines starting with space in the history.
# See bash(1) for more options
HISTCONTROL=ignoreboth

# append to the history file, don't overwrite it
shopt -s histappend

# for setting history length see HISTSIZE and HISTFILESIZE in bash(1)
HISTSIZE=1000
HISTFILESIZE=2000

# check the window size after each command and, if necessary,
# update the values of LINES and COLUMNS.
shopt -s checkwinsize

# If set, the pattern "**" used in a pathname expansion context will
# match all files and zero or more directories and subdirectories.
#shopt -s globstar

# make less more friendly for non-text input files, see lesspipe(1)
[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

# set variable identifying the chroot you work in (used in the prompt below)
if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

# set a fancy prompt (non-color, unless we know we "want" color)
case "$TERM" in
    xterm-color|*-256color) color_prompt=yes;;
esac

# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes

if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
	# We have color support; assume it's compliant with Ecma-48
	# (ISO/IEC-6429). (Lack of such support is extremely rare, and such
	# a case would tend to support setf rather than setaf.)
	color_prompt=yes
    else
	color_prompt=
    fi
fi

if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

# If this is an xterm set the title to user@host:dir
case "$TERM" in
xterm*|rxvt*)
    PS1="\[\e]0;${debian_chroot:+($debian_chroot)}\u@\h: \w\a\]$PS1"
    ;;
*)
    ;;
esac

# enable color support of ls and also add handy aliases
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    #alias dir='dir --color=auto'
    #alias vdir='vdir --color=auto'

    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi

# colored GCC warnings and errors
#export GCC_COLORS='error=01;31:warning=01;35:note=01;36:caret=01;32:locus=01:quote=01'

# changing colors for `ls`: https://askubuntu.com/questions/466198/how-do-i-change-the-color-for-directories-with-ls-in-the-console
LS_COLORS=$LS_COLORS:'di=0;93:' ; export LS_COLORS

# some more ls aliases
alias ll='ls -alFv'
alias la='ls -A'
alias l='ls -CF'

# Add an "alert" alias for long running commands.  Use like so:
#   sleep 10; alert
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'

# trying to fix autocomplete of filenames after arg flags in custom python script
complete -D -o default

# Alias definitions.
# You may want to put all your additions into a separate file like
# ~/.bash_aliases, instead of adding them here directly.
# See /usr/share/doc/bash-doc/examples in the bash-doc package.

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

# enable programmable completion features (you don't need to enable
# this, if it's already enabled in /etc/bash.bashrc and /etc/profile
# sources /etc/bash.bashrc).
if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

export LESS="-SR"  # turns off line wrapping in less
# export PYTHONPATH=$PYTHONPATH:/home/wesley/programming  # this is a bad idea if using python outside of programming dir, it will try to look for libraries there
export GOPATH=$HOME/gopath:$HOME/gopath/bin:/usr/local/go/bin
export ANDROIDSTUDIOPATH=$HOME/android-studio:$HOME/android-studio/bin
export GIO_EXTRA_MODULES=/usr/lib/x86_64-linux-gnu/gio/modules/  # https://stackoverflow.com/questions/44934641/
export MAVENPATH=$HOME/apache-maven-3.8.4/bin
export POTRACE=$HOME/potrace-1.16.linux-x86_64/potrace  # for FontForge autotrace plugin
export PATH=~/.npm-global/bin:$PATH
export PATH=$PATH:$PYTHONPATH:$GOPATH:$ANDROIDSTUDIOPATH:$MAVENPATH
export CLASSPATH=".:/usr/local/lib/antlr-4.13.1-complete.jar:$CLASSPATH"  # for getting ANTLR java to compile so I can use grun to visualize parse trees
export PATH=/usr/local/texlive/2022/bin/x86_64-linux:$PATH
export INFOPATH=$INFOPATH:/usr/local/texlive/2022/texmf-dist/doc/info
export MANPATH=$MANPATH:/usr/local/texlive/2022/texmf-dist/doc/man

# python3 virtualenvwrapper
export WORKON_HOME=~/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=~/.local/bin/virtualenv
source ~/.local/bin/virtualenvwrapper.sh

# nvm (Node.js Version Manager)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion

# Install Ruby Gems to ~/gems
export GEM_HOME="$HOME/gems"
export PATH="$HOME/gems/bin:$PATH"

alias vbrc="vim ~/.bashrc"
alias sbrc="source ~/.bashrc"
function gbrc() { 
  grep -i "$1" ~/.bashrc 
}
function pygrep() {
  grep --include \*.py $@
}
# alias python2="/usr/bin/python"
alias python="python3"
# alias pip2="pip"
alias pip="pip3"
alias battery="upower -i /org/freedesktop/UPower/devices/battery_BAT0 | grep \"state\|percentage\""
alias chrome="google-chrome-stable"
alias skype="pulseaudio -k; PULSE_LATENCY_MSEC=90 skypeforlinux"
alias minecraft="java -jar ~/.minecraft/launcher.jar"
alias pclear="python -c \"print('\n'*1000)\" && clear"
alias elan="sudo /opt/ELAN_5-9/ELAN"
alias j="jobs"
alias sound="alsactl restore"
alias xo="xdg-open"
alias ff="firefox"
alias cr="chrome"
alias lynx="lynx -accept_all_cookies"
alias kgs="java -jar ~/Desktop/Learning/Games/Go/cgoban.jar &"
alias sublime="/opt/sublime_text/sublime_text &"
alias ipa="vim /usr/share/kmfl/IPA14.kmn"
alias zotero="/opt/zotero/zotero &"
alias flex="/usr/bin/fieldworks-flex"
alias antlr4='java -Xmx500M -cp "~/antlr-4.13.1-complete.jar:$CLASSPATH" org.antlr.v4.Tool'
alias grun='echo "do not use; use ParsingDebugging.py which I wrote instead"' #'java -Xmx500M -cp "/usr/local/lib/antlr-4.7.1-complete.jar:$CLASSPATH" org.antlr.v4.gui.TestRig'
alias antlrworks='java -jar ~/antlrworks-1.5.2-complete.jar'
alias grepdocx="/home/wesley/grepdocx.sh"
alias rpdf="python /home/wesley/programming/ReadRandomPdf.py"
alias amongus="sudo wine .steam/steam/steamapps/common/Among\ Us/Among\ Us.exe"
alias synctime="sudo tlsdate -s -H mail.google.com"
alias gitsize="git status --porcelain | sed 's/^...//g;s/\"//g' | xargs -d '\n' -I {} du -h {} | sort -h"
alias gephi="$HOME/gephi-0.10.1/bin/gephi"  # graph visualization program
alias tmux="TERM=screen-256color-bce tmux"
alias ft="find . -printf '%T@ %Tc %p\\n' | sort -n"
alias dry="python /home/kuhron/drybones/src/drybones/main.py"
alias dark='gsettings set org.gnome.desktop.interface gtk-theme Adwaita-dark && gsettings set org.gnome.desktop.interface color-scheme prefer-dark'
alias light='gsettings set org.gnome.desktop.interface gtk-theme Adwaita && gsettings set org.gnome.desktop.interface color-scheme prefer-light'
alias lsg='/home/kuhron/lsg'  # ls with git repo information
alias k1='kill %1'


function nwc() {
    # alias nwc="killall timidity ; timidity -iA -B2,8 -Os & wine \"/home/wesley/.wine/drive_c/Program Files (x86)/Noteworthy Software/NoteWorthy Composer 2 Demo/NWC2Demo.exe\" &"  # if you killall & timidity then it will kill the one you just started  # this is for the demo version before I bought a license
    # sudo killall timidity
    # timid
    wine "/home/wesley/.wine/drive_c/Program Files (x86)/Noteworthy Software/NoteWorthy Composer 2/NWC2.exe" &
}

function timid() {
    # timidity -iA -B2,8 -Os &  # -Os option causes segfault on Ubuntu 22
    timidity -iA -B2,8 &
}

function midtomp3() { echo "this is broken because I don't know how to embed sed result as output filename"; timidity "$1" -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 64k $(sed -e "s/mid$/mp3/" "$1") ;}

function truncate() { cut -c 1-$(tput cols) $1 ;}
function psg() { ps aux | grep $1 | grep -v grep | truncate ;}
# function null() { "$@" &> /dev/null ;}  # doesn't work
function aco() {  # play acoma audio files matching string
    b=$(
        a=$(echo $1 | sed -e "s/\-[0-9]\./\./g;s/\.mp3//g"); 
        find . | grep -i $a | grep -v mp3);
    echo $b | sed -e "s/ /\n/g"; echo "$b" | xargs -I % sh -c '{ aplay %; sleep 1; }';
}
function bright() { xrandr --output $(xrandr | grep " connected" | cut -f1 -d " ") --brightness $1 ;}

function tab4() {
    for arg in "$@"; do  # https://mywiki.wooledge.org/BashPitfalls (#24)
        echo $arg
        expand -t 4 $arg > "_TEMP_TAB_TO_SPACE_"$arg
        mv "_TEMP_TAB_TO_SPACE_"$arg $arg
    done
}

function cdiff() { /usr/bin/diff -u $1 $2 | colordiff | less -R ;}  # https://stackoverflow.com/a/20288437/7376935
alias diff="cdiff"  # this way it's unambiguous what I want when I type "diff", it goes to the custom function

function setmainfont() {
    dconf write /org/gnome/desktop/interface/monospace-font-name "'Ubuntu Mono 15'"
}

# nox, for obscuring entire terminal into a font unreadable by others, usually for use with lynx browser
function nox() {
    : # nop
    clear
    echo "Choose Nox mode: b,m"
    read mode
    case "$mode" in
        "b")
            dconf write /org/gnome/desktop/interface/monospace-font-name "'Braille Normal 800 11'" ;;
        "m")
            dconf write /org/gnome/desktop/interface/monospace-font-name "'MorseCodeScript 9'" ;;
        *)
            echo "invalid mode" ;;
    esac
}
function denox() {
    : # nop
    clear
    setmainfont
}
#function g() { git add -A; git commit -m "$1"; git push; }  # doesn't work

function base() { 
    case "$2" in
        "10")
            s2="" ;;
        *)
            s2="ibase=$2; " ;;
    esac
    case "$3" in
        "10")
            s3="" ;;
        *)
            s3="obase=$3; " ;;
    esac
    echo "converting $1 from base $2 to base $3"
    echo "$s2$s3$1" | bc
}

function all-webp-to-png() {
    find -name "*.webp" | xargs -d "\n" -I "{}" python3 ~/programming/WebpToPng.py "{}"
}

function find-images() {
    # https://stackoverflow.com/questions/16758105/list-all-graphic-image-files-with-find
    find $1 -type f -exec file --mime-type {} \+ | awk -F: '{if ($2 ~/image\//) print $1}'
}

function rimg() {
    find-images $1 | shuf | head -n 1 | xargs -d "\n" -I "{}" xdg-open "{}"
}

# COD MW2 controls
function cod_without_mouse() {
    cp ~/Games/custom_configs/mw2_controls_without_mouse.cfg ~/.steam/steam/steamapps/common/Call\ of\ Duty\ Modern\ Warfare\ 2/players/config_mp.cfg
}
function cod_with_mouse() {
    cp ~/Games/custom_configs/mw2_controls_with_mouse.cfg ~/.steam/steam/steamapps/common/Call\ of\ Duty\ Modern\ Warfare\ 2/players/config_mp.cfg
}

# write symlink source and destination paths to a text file, for backing up to EHD filesystem that doesn't support symlinks
function echo_symlink_source_and_destination() {
    dest=$(readlink -f "$1")
    src="$1"
    echo "\"""$src""\" -> \"""$dest""\""
}
export -f echo_symlink_source_and_destination
function echo_symlink_sources_and_destinations() {
    find "$1" -type l | xargs -d "\n" -I "{}" bash -c 'echo_symlink_source_and_destination "{}"'
}

# clock format, for if you accidentally use the GUI to change date/time settings and lose the custom format
gsettings set org.gnome.desktop.interface clock-show-seconds true
# - in gnome, com.canonical.indicator.datetime doesn't work, need to use Gnome Shell Extension "Panel Date Format", but keeping these lines here in case of future switch back to Unity, and so I still have the format string
# gsettings set com.canonical.indicator.datetime time-format "'custom'"
# gsettings set com.canonical.indicator.datetime custom-time-format "'%Y-%m-%d %H:%M:%S %Z  ||  %w  %j  %s'"

gsettings set org.gnome.desktop.peripherals.touchpad disable-while-typing false

# make desktop background plain black
gsettings set org.gnome.desktop.background picture-uri ''
gsettings set org.gnome.desktop.background picture-uri-dark ''
gsettings set org.gnome.desktop.background primary-color 'rgb(0, 0, 0)'

setmainfont


