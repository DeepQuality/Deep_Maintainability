//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package ast.parser;

import java.util.HashMap;
import java.util.Scanner;
import java.util.regex.Pattern;

import treedist.Tree;

public class AST implements Tree {
    private AST.Node root;
    private HashMap<Integer, AST.Node> nodes = new HashMap();

    public static AST parseDot(String dot) {
        Scanner sc = new Scanner(dot);
        AST ast = new AST();
        String line = "";

        while(true) {
            while(true) {
                while(sc.hasNextLine()) {
                    line = line + sc.nextLine().trim();
                    if (!line.startsWith("digraph") && !line.startsWith("}") && line.length() != 0) {
                        if (Pattern.matches("n\\d+ -> n\\d+;", line)) {
                            String[] ns = line.split("->");
                            ns[0] = ns[0].trim();
                            ns[1] = ns[1].trim();
                            int s = Integer.parseInt(ns[0].substring(1));
                            int t = Integer.parseInt(ns[1].substring(1, ns[1].length() - 1));
                            ast.addEdge(s, t);
                            line = "";
                        } else if (Pattern.matches("n\\d+ \\[label=.+\\];", line)) {
                            int id = Integer.parseInt(line.substring(0, line.indexOf(" ")).substring(1).trim());
                            String label = line.substring(line.indexOf(" ") + 1);
                            label = label.substring(8, label.length() - 3);
                            ast.add(id, label);
                            if (label.startsWith("root")) {
                                ast.setRoot(id);
                            }

                            line = "";
                        }
                    } else {
                        line = "";
                    }
                }

                return ast;
            }
        }
    }

    private AST() {
    }

    public String getNodeLabel(int nodeid) {
        AST.Node node = (AST.Node)this.nodes.get(nodeid);
        return node == null ? null : node.label;
    }

    public int children(int nodeid) {
        AST.Node node = (AST.Node)this.nodes.get(nodeid);
        if (node == null) {
            return 0;
        } else {
            int cnt = 0;

            for(AST.Node ch = node.firstChild; ch != null; ++cnt) {
                ch = ch.nextSibling;
            }

            return cnt;
        }
    }

    public void setRoot(int id) {
        AST.Node node = (AST.Node)this.nodes.get(id);
        if (node != null) {
            this.root = node;
        }
    }

    public void add(int id, String label) {
        if (label.startsWith("value=")) {
            String content = label.substring(7, label.length() - 1);
            content = content.replaceAll("\\'", "\\\\'");
            label = "value='" + content + "'";
        }

        AST.Node node = new AST.Node(id, label);
        this.nodes.put(id, node);
    }

    public void addEdge(int p, int c) {
        AST.Node parent = (AST.Node)this.nodes.get(p);
        AST.Node child = (AST.Node)this.nodes.get(c);
        parent.add(child);
    }

    public void addEdgeAsFirstChild(int p, int c) {
        AST.Node parent = (AST.Node)this.nodes.get(p);
        AST.Node child = (AST.Node)this.nodes.get(c);
        parent.addAsFirstChild(child);
    }

    public int getRoot() {
        return this.root.id;
    }

    public int getFirstChild(int nodeId) {
        AST.Node node = (AST.Node)this.nodes.get(nodeId);
        if (node == null) {
            return -1;
        } else {
            return node.firstChild == null ? -1 : node.firstChild.id;
        }
    }

    public int getLastChild(int nodeId) {
        int ch = this.getFirstChild(nodeId);
        if (ch == -1) {
            return -1;
        } else {
            for(int nch = this.getNextSibling(ch); nch != -1; nch = this.getNextSibling(nch)) {
                ch = nch;
            }

            return ch;
        }
    }

    public int getNextSibling(int nodeId) {
        AST.Node node = (AST.Node)this.nodes.get(nodeId);
        if (node == null) {
            return -1;
        } else {
            return node.nextSibling == null ? -1 : node.nextSibling.id;
        }
    }

    public int getParent(int nodeId) {
        AST.Node node = (AST.Node)this.nodes.get(nodeId);
        if (node == null) {
            return -1;
        } else {
            return node.parent == null ? -1 : node.parent.id;
        }
    }

    public int size() {
        return this.nodes.size();
    }

    public void print() {
        this.root.print("");
    }

    public void print_s(StringBuilder sb) {
        this.root.print_n(sb);
        this.root.print_e(sb);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.size() + "\n");
        this.print_s(sb);
        return sb.toString();
    }

    private class Node {
        int id;
        String label;
        AST.Node parent;
        AST.Node firstChild;
        AST.Node lastChild;
        AST.Node nextSibling;

        Node(int id, String label) {
            this.id = id;
            this.label = label;
        }

        void add(AST.Node ch) {
            ch.parent = this;
            if (this.firstChild == null) {
                this.firstChild = ch;
            }

            if (this.lastChild != null) {
                this.lastChild.nextSibling = ch;
            }

            this.lastChild = ch;
        }

        void addAsFirstChild(AST.Node ch) {
            ch.parent = this;
            ch.nextSibling = this.firstChild;
            this.firstChild = ch;
            if (this.lastChild == null) {
                this.lastChild = ch;
            }

        }

        void print(String indent) {
            System.out.println(indent + this.id + ": " + this.label);

            for(AST.Node node = this.firstChild; node != null; node = node.nextSibling) {
                node.print(indent + "-");
            }

        }

        public void print_n(StringBuilder sb) {
            sb.append(this.id + " " + this.label + "\n");

            for(AST.Node node = this.firstChild; node != null; node = node.nextSibling) {
                node.print_n(sb);
            }

        }

        public void print_e(StringBuilder sb) {
            for(AST.Node node = this.firstChild; node != null; node = node.nextSibling) {
                sb.append(this.id + " " + node.id + "\n");
                node.print_e(sb);
            }

        }
    }
}
