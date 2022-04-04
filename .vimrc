set expandtab
set tabstop=4
set shiftwidth=4  " indentation amount
set number

" indent xml: https://stackoverflow.com/questions/21408222/vim-indent-xml-file/32405832
:set equalprg=xmllint\ --format\ -

" Allow saving of files as sudo when I forgot to start vim using sudo.
" https://stackoverflow.com/questions/2600783/how-does-the-vim-write-with-sudo-trick-work
cmap w!! w !sudo tee > /dev/null %

" show whitespace as chars except spaces
:set listchars=tab:——,trail:·,extends:>,precedes:<
" WKJ removed eol char"
:set list

