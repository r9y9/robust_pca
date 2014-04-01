# Robust Principal Component Analysis
--------------------------------------

[P.S Huang, S.D. Chen, P. Smaragdis, and M. HasegawaJohnson, "Singing-voice separation from monaural recordings using robust principal component analysis", Proc. ICASSP 2012.](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202012/pdfs/0000057.pdf)

Matlabコードは公開されてるけど使いにくいのでベースだけC++で書きなおした版

本家のデモとかこのへんにある -> [https://sites.google.com/site/singingvoiceseparationrpca/](https://sites.google.com/site/singingvoiceseparationrpca/)



面白い方法だけど、行列がデカイと計算もけっこう時間かかる

## メモ

* RedSVDとか使って高速化しようと思ったけど途中でやめた
* ↑に上げた論文だけじゃわからなくて色々調べて実装した（曖昧）