" Vim color file

" cool help screens
" :he group-name
" :he highlight-groups
" :he cterm-colors

set background=dark
if version > 580
    " no guarantees for version 5.8 and below, but this makes it stop
    " complaining
    hi clear
    if exists("syntax_on")
	syntax reset
    endif
endif
let g:colors_name="volcano"

hi Normal	guifg=Yellow1 guibg=grey20

" highlight groups
hi Cursor	guibg=HotPink guifg=slategrey
"hi CursorIM
"hi Directory
"hi DiffAdd
"hi DiffChange
"hi DiffDelete
"hi DiffText
"hi ErrorMsg
hi VertSplit	guibg=orange guifg=grey50 gui=none
hi Folded	guibg=grey30 guifg=gold
hi FoldColumn	guibg=grey30 guifg=LightSalmon
hi IncSearch	guifg=slategrey guibg=HotPink
hi LineNr   guifg=orange
hi ModeMsg	guifg=yellow1
hi MoreMsg	guifg=orange
hi NonText	guifg=tan1 guibg=grey30
hi Question	guifg=gold1
hi Search	guibg=peru guifg=wheat
hi SpecialKey	guifg=VioletRed1
hi StatusLine	guibg=orange guifg=black gui=none
hi StatusLineNC	guibg=orange guifg=grey50 gui=none
hi Title	guifg=indianred
hi Visual	gui=none guifg=HotPink guibg=tomato
"hi VisualNOS
hi WarningMsg	guifg=salmon
"hi WildMenu
"hi Menu
"hi Scrollbar
"hi Tooltip

" syntax highlighting groups
hi Comment	guifg=OrangeRed
hi Constant	guifg=grey90
hi Identifier	guifg=goldenrod
hi Statement	guifg=red
hi PreProc	guifg=indianred
hi Type		guifg=maroon1
hi Special	guifg=salmon
"hi Underlined
hi Ignore	guifg=grey80
"hi Error
hi Todo		guifg=yellow2 guibg=red

" color terminal definitions
hi SpecialKey	ctermfg=yellow
hi NonText	cterm=bold ctermfg=red
hi Directory	ctermfg=darkcyan
hi ErrorMsg	cterm=bold ctermfg=yellow ctermbg=darkred
hi IncSearch	cterm=NONE ctermfg=yellow ctermbg=red
hi Search	cterm=NONE ctermfg=white ctermbg=red
hi MoreMsg	ctermfg=yellow
hi ModeMsg	cterm=NONE ctermfg=red
hi LineNr	ctermfg=yellow
hi Question	ctermfg=yellow
hi StatusLine	cterm=bold,reverse
hi StatusLineNC cterm=reverse
hi VertSplit	cterm=reverse
hi Title	ctermfg=yellow
hi Visual	cterm=reverse
hi VisualNOS	cterm=bold,underline
hi WarningMsg	ctermfg=red
hi WildMenu	ctermfg=white ctermbg=red
hi Folded	ctermfg=darkgrey ctermbg=NONE
hi FoldColumn	ctermfg=darkgrey ctermbg=NONE
hi DiffAdd	ctermbg=green
hi DiffChange	ctermbg=yellow
hi DiffDelete	cterm=bold ctermfg=red ctermbg=black
hi DiffText	cterm=bold ctermbg=white
hi Comment	ctermfg=yellow
hi Constant	ctermfg=red
hi Special	ctermfg=lightblue
hi Identifier	ctermfg=red
hi Statement	ctermfg=lightyellow
hi PreProc	ctermfg=lightyellow
hi Type		ctermfg=lightyellow
hi Underlined	cterm=underline ctermfg=red
hi Ignore	cterm=bold ctermfg=grey
hi Ignore	ctermfg=grey
hi Error	cterm=bold ctermfg=yellow ctermbg=red


"vim: sw=4
