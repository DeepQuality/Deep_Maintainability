package main;

import main.encodings.IdentifierEncoding;
import org.junit.Test;
import main.util.IdentifierSpliter;

import java.util.ArrayList;

import static org.junit.Assert.*;

public class IdentifierEncodingTest {

    @Test
    public void isType1() {
        // 1 全小写  2 首字母大写  3 全大写  4 其他
        boolean abd = IdentifierEncoding.isType1("abd");
        assertTrue(abd);
        abd = IdentifierEncoding.isType1("Abd");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType1("AASA");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType1("AfsA");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType1("f");
        assertTrue(abd);
        abd = IdentifierEncoding.isType1("F");
        assertTrue(!abd);
    }

    @Test
    public void isType2() {
        boolean abd = IdentifierEncoding.isType2("abd");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType2("Abd");
        assertTrue(abd);
        abd = IdentifierEncoding.isType2("AASA");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType2("AfsA");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType2("f");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType2("F");
        assertTrue(!abd);
    }

    @Test
    public void isType3() {
        boolean abd = IdentifierEncoding.isType3("abd");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType3("Abd");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType3("AASA");
        assertTrue(abd);
        abd = IdentifierEncoding.isType3("AfsA");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType3("f");
        assertTrue(!abd);
        abd = IdentifierEncoding.isType3("F");
        assertTrue(abd);
    }


    @Test
    public void split() {
        ArrayList<String> abcAfeGew = IdentifierSpliter.splitIdentifier("AbcAfeGew");
        assertTrue(isEqual(abcAfeGew, new String[]{"Abc", "Afe", "Gew"}) );
        abcAfeGew = IdentifierSpliter.splitIdentifier("abcAfeGew");
        assertTrue(isEqual(abcAfeGew, new String[]{"abc", "Afe", "Gew"}) );
        abcAfeGew = IdentifierSpliter.splitIdentifier("aAA");
        assertTrue(isEqual(abcAfeGew, new String[]{"a", "AA"}) );
        abcAfeGew = IdentifierSpliter.splitIdentifier("AAA");
        assertTrue(isEqual(abcAfeGew, new String[]{"AAA"}) );
        abcAfeGew = IdentifierSpliter.splitIdentifier("asfEEFAfe");
        assertTrue(isEqual(abcAfeGew, new String[]{"asf", "EEF", "Afe"}) );

        abcAfeGew = IdentifierSpliter.splitIdentifier("$abcA_2feGe3w");
        assertTrue(isEqual(abcAfeGew, new String[]{"abc", "A", "fe", "Ge", "w"}) );
    }

    @Test
    public void splitBigLetter() {
        ArrayList<String> abcAfeGew = IdentifierSpliter.splitBigLetter("AbcAfeGew");
        assertTrue(isEqual(abcAfeGew, new String[]{"Abc", "Afe", "Gew"}) );
        abcAfeGew = IdentifierSpliter.splitBigLetter("abcAfeGew");
        assertTrue(isEqual(abcAfeGew, new String[]{"abc", "Afe", "Gew"}) );
        abcAfeGew = IdentifierSpliter.splitBigLetter("aAA");
        assertTrue(isEqual(abcAfeGew, new String[]{"a", "AA"}) );
        abcAfeGew = IdentifierSpliter.splitBigLetter("AAA");
        assertTrue(isEqual(abcAfeGew, new String[]{"AAA"}) );
        abcAfeGew = IdentifierSpliter.splitBigLetter("asfEEFAfe");
        assertTrue(isEqual(abcAfeGew, new String[]{"asf", "EEF", "Afe"}) );
    }

    public boolean isEqual(ArrayList<String> list, String[] array) {
        if (list.size() != array.length) {
            return false;
        }
        for (int i = 0; i < list.size(); i++) {
            if (!list.get(i).equals(array[i])) {
                return false;
            }
        }
        return true;
    }
}