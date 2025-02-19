"use strict";(self.webpackChunknotes_template=self.webpackChunknotes_template||[]).push([[6153],{4981:(s,a,e)=>{e.r(a),e.d(a,{assets:()=>t,contentTitle:()=>c,default:()=>x,frontMatter:()=>m,metadata:()=>n,toc:()=>r});const n=JSON.parse('{"id":"binary-classification/binary-classification","title":"Binary Classification","description":"Binary Classification solves the problem,","source":"@site/docs/binary-classification/binary-classification.mdx","sourceDirName":"binary-classification","slug":"/binary-classification/","permalink":"/classical-ml-basics/docs/binary-classification/","draft":false,"unlisted":false,"editUrl":"https://github.com/finger-bone/classical-ml-basics/blob/main/docs/binary-classification/binary-classification.mdx","tags":[],"version":"current","sidebarPosition":1,"frontMatter":{"sidebar_position":1},"sidebar":"tutorialSidebar","previous":{"title":"Binary Classification","permalink":"/classical-ml-basics/docs/category/binary-classification"},"next":{"title":"Support Vector Machine","permalink":"/classical-ml-basics/docs/binary-classification/support-vector-machine"}}');var l=e(4848),i=e(8453);const m={sidebar_position:1},c="Binary Classification",t={},r=[];function h(s){const a={annotation:"annotation",h1:"h1",header:"header",math:"math",mi:"mi",mn:"mn",mo:"mo",mover:"mover",mrow:"mrow",msub:"msub",msup:"msup",p:"p",semantics:"semantics",span:"span",...(0,i.R)(),...s.components};return(0,l.jsxs)(l.Fragment,{children:[(0,l.jsx)(a.header,{children:(0,l.jsx)(a.h1,{id:"binary-classification",children:"Binary Classification"})}),"\n",(0,l.jsx)(a.p,{children:"Binary Classification solves the problem,"}),"\n",(0,l.jsx)(a.span,{className:"katex-display",children:(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsxs)(a.mrow,{children:[(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"x"}),(0,l.jsx)(a.mi,{children:"i"})]}),(0,l.jsx)(a.mo,{children:"\u2208"}),(0,l.jsxs)(a.msup,{children:[(0,l.jsx)(a.mi,{mathvariant:"double-struck",children:"R"}),(0,l.jsx)(a.mi,{children:"n"})]}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"y"}),(0,l.jsx)(a.mi,{children:"i"})]}),(0,l.jsx)(a.mo,{children:"\u2208"}),(0,l.jsx)(a.mi,{mathvariant:"double-struck",children:"B"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mi,{children:"i"}),(0,l.jsx)(a.mo,{children:"="}),(0,l.jsx)(a.mn,{children:"1"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mn,{children:"2"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mo,{children:"\u2026"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mi,{children:"n"})]}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"x_i \\in \\mathbb{R}^n, y_i \\in \\mathbb{B}, i = 1, 2, \\ldots, n"})]})})}),(0,l.jsxs)(a.span,{className:"katex-html","aria-hidden":"true",children:[(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.6891em",verticalAlign:"-0.15em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3117em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mathnormal mtight",children:"i"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"\u2208"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.9088em",verticalAlign:"-0.1944em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathbb",children:"R"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsx)(a.span,{className:"vlist-t",children:(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.7144em"},children:(0,l.jsxs)(a.span,{style:{top:"-3.113em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mathnormal mtight",children:"n"})})]})})})})})]}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3117em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mathnormal mtight",children:"i"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"\u2208"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.8833em",verticalAlign:"-0.1944em"}}),(0,l.jsx)(a.span,{className:"mord mathbb",children:"B"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mord mathnormal",children:"i"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"="}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.8389em",verticalAlign:"-0.1944em"}}),(0,l.jsx)(a.span,{className:"mord",children:"1"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mord",children:"2"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"minner",children:"\u2026"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mord mathnormal",children:"n"})]})]})]})}),"\n",(0,l.jsxs)(a.p,{children:["Where ",(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsxs)(a.mrow,{children:[(0,l.jsx)(a.mi,{mathvariant:"double-struck",children:"B"}),(0,l.jsx)(a.mo,{children:"="}),(0,l.jsx)(a.mo,{stretchy:"false",children:"{"}),(0,l.jsx)(a.mn,{children:"0"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mn,{children:"1"}),(0,l.jsx)(a.mo,{stretchy:"false",children:"}"})]}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"\\mathbb{B} = \\{0,1\\}"})]})})}),(0,l.jsxs)(a.span,{className:"katex-html","aria-hidden":"true",children:[(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.6889em"}}),(0,l.jsx)(a.span,{className:"mord mathbb",children:"B"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"="}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,l.jsx)(a.span,{className:"mopen",children:"{"}),(0,l.jsx)(a.span,{className:"mord",children:"0"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mord",children:"1"}),(0,l.jsx)(a.span,{className:"mclose",children:"}"})]})]})]})," or sometimes, ",(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsxs)(a.mrow,{children:[(0,l.jsx)(a.mi,{mathvariant:"double-struck",children:"B"}),(0,l.jsx)(a.mo,{children:"="}),(0,l.jsx)(a.mo,{stretchy:"false",children:"{"}),(0,l.jsx)(a.mo,{children:"\u2212"}),(0,l.jsx)(a.mn,{children:"1"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mn,{children:"1"}),(0,l.jsx)(a.mo,{stretchy:"false",children:"}"})]}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"\\mathbb{B} = \\{-1,1\\}"})]})})}),(0,l.jsxs)(a.span,{className:"katex-html","aria-hidden":"true",children:[(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.6889em"}}),(0,l.jsx)(a.span,{className:"mord mathbb",children:"B"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"="}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,l.jsx)(a.span,{className:"mopen",children:"{"}),(0,l.jsx)(a.span,{className:"mord",children:"\u2212"}),(0,l.jsx)(a.span,{className:"mord",children:"1"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mord",children:"1"}),(0,l.jsx)(a.span,{className:"mclose",children:"}"})]})]})]}),"."]}),"\n",(0,l.jsx)(a.p,{children:"And a train dataset,"}),"\n",(0,l.jsx)(a.span,{className:"katex-display",children:(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsxs)(a.mrow,{children:[(0,l.jsx)(a.mi,{mathvariant:"script",children:"D"}),(0,l.jsx)(a.mo,{children:"="}),(0,l.jsx)(a.mo,{stretchy:"false",children:"{"}),(0,l.jsx)(a.mo,{stretchy:"false",children:"("}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"x"}),(0,l.jsx)(a.mn,{children:"1"})]}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"y"}),(0,l.jsx)(a.mn,{children:"1"})]}),(0,l.jsx)(a.mo,{stretchy:"false",children:")"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mo,{stretchy:"false",children:"("}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"x"}),(0,l.jsx)(a.mn,{children:"2"})]}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"y"}),(0,l.jsx)(a.mn,{children:"2"})]}),(0,l.jsx)(a.mo,{stretchy:"false",children:")"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mo,{children:"\u2026"}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsx)(a.mo,{stretchy:"false",children:"("}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"x"}),(0,l.jsx)(a.mi,{children:"n"})]}),(0,l.jsx)(a.mo,{separator:"true",children:","}),(0,l.jsxs)(a.msub,{children:[(0,l.jsx)(a.mi,{children:"y"}),(0,l.jsx)(a.mi,{children:"n"})]}),(0,l.jsx)(a.mo,{stretchy:"false",children:")"}),(0,l.jsx)(a.mo,{stretchy:"false",children:"}"})]}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"\\mathcal{D} = \\{(x_1, y_1), (x_2, y_2), \\ldots, (x_n, y_n)\\}"})]})})}),(0,l.jsxs)(a.span,{className:"katex-html","aria-hidden":"true",children:[(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.6833em"}}),(0,l.jsx)(a.span,{className:"mord mathcal",style:{marginRight:"0.02778em"},children:"D"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"="}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,l.jsx)(a.span,{className:"mopen",children:"{("}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3011em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mtight",children:"1"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3011em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mtight",children:"1"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mclose",children:")"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mopen",children:"("}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3011em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mtight",children:"2"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.3011em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mtight",children:"2"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mclose",children:")"}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"minner",children:"\u2026"}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsx)(a.span,{className:"mopen",children:"("}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"0em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mathnormal mtight",children:"n"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mpunct",children:","}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.1667em"}}),(0,l.jsxs)(a.span,{className:"mord",children:[(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"}),(0,l.jsx)(a.span,{className:"msupsub",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.1514em"},children:(0,l.jsxs)(a.span,{style:{top:"-2.55em",marginLeft:"-0.0359em",marginRight:"0.05em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"2.7em"}}),(0,l.jsx)(a.span,{className:"sizing reset-size6 size3 mtight",children:(0,l.jsx)(a.span,{className:"mord mathnormal mtight",children:"n"})})]})}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.15em"},children:(0,l.jsx)(a.span,{})})})]})})]}),(0,l.jsx)(a.span,{className:"mclose",children:")}"})]})]})]})}),"\n",(0,l.jsxs)(a.p,{children:["Finds the best fit ",(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsx)(a.mrow,{children:(0,l.jsx)(a.mi,{children:"f"})}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"f"})]})})}),(0,l.jsx)(a.span,{className:"katex-html","aria-hidden":"true",children:(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.8889em",verticalAlign:"-0.1944em"}}),(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.10764em"},children:"f"})]})})]})," for,"]}),"\n",(0,l.jsx)(a.span,{className:"katex-display",children:(0,l.jsxs)(a.span,{className:"katex",children:[(0,l.jsx)(a.span,{className:"katex-mathml",children:(0,l.jsx)(a.math,{xmlns:"http://www.w3.org/1998/Math/MathML",display:"block",children:(0,l.jsxs)(a.semantics,{children:[(0,l.jsxs)(a.mrow,{children:[(0,l.jsxs)(a.mover,{accent:"true",children:[(0,l.jsx)(a.mi,{children:"y"}),(0,l.jsx)(a.mo,{children:"^"})]}),(0,l.jsx)(a.mo,{children:"="}),(0,l.jsx)(a.mi,{children:"f"}),(0,l.jsx)(a.mo,{stretchy:"false",children:"("}),(0,l.jsx)(a.mi,{children:"x"}),(0,l.jsx)(a.mo,{stretchy:"false",children:")"})]}),(0,l.jsx)(a.annotation,{encoding:"application/x-tex",children:"\\hat{y} = f(x)"})]})})}),(0,l.jsxs)(a.span,{className:"katex-html","aria-hidden":"true",children:[(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"0.8889em",verticalAlign:"-0.1944em"}}),(0,l.jsx)(a.span,{className:"mord accent",children:(0,l.jsxs)(a.span,{className:"vlist-t vlist-t2",children:[(0,l.jsxs)(a.span,{className:"vlist-r",children:[(0,l.jsxs)(a.span,{className:"vlist",style:{height:"0.6944em"},children:[(0,l.jsxs)(a.span,{style:{top:"-3em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"3em"}}),(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.03588em"},children:"y"})]}),(0,l.jsxs)(a.span,{style:{top:"-3em"},children:[(0,l.jsx)(a.span,{className:"pstrut",style:{height:"3em"}}),(0,l.jsx)(a.span,{className:"accent-body",style:{left:"-0.1944em"},children:(0,l.jsx)(a.span,{className:"mord",children:"^"})})]})]}),(0,l.jsx)(a.span,{className:"vlist-s",children:"\u200b"})]}),(0,l.jsx)(a.span,{className:"vlist-r",children:(0,l.jsx)(a.span,{className:"vlist",style:{height:"0.1944em"},children:(0,l.jsx)(a.span,{})})})]})}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}}),(0,l.jsx)(a.span,{className:"mrel",children:"="}),(0,l.jsx)(a.span,{className:"mspace",style:{marginRight:"0.2778em"}})]}),(0,l.jsxs)(a.span,{className:"base",children:[(0,l.jsx)(a.span,{className:"strut",style:{height:"1em",verticalAlign:"-0.25em"}}),(0,l.jsx)(a.span,{className:"mord mathnormal",style:{marginRight:"0.10764em"},children:"f"}),(0,l.jsx)(a.span,{className:"mopen",children:"("}),(0,l.jsx)(a.span,{className:"mord mathnormal",children:"x"}),(0,l.jsx)(a.span,{className:"mclose",children:")"})]})]})]})}),"\n",(0,l.jsx)(a.p,{children:"By best fit, we typically mean to minimize a loss value. Depending on the algorithm, the loss function can be different."})]})}function x(s={}){const{wrapper:a}={...(0,i.R)(),...s.components};return a?(0,l.jsx)(a,{...s,children:(0,l.jsx)(h,{...s})}):h(s)}}}]);