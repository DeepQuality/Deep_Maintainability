package main.util;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashSet;

public class Dic {

    public static HashSet<String> englishDicHashSet = new HashSet<>();
    static {
        // https://github.com/dwyl/english-words/blob/master/words.txt
        try {
            InputStream resourceAsStream = Dic.class.getClassLoader().getResourceAsStream("words.txt");
            assert resourceAsStream != null;
            BufferedReader reader = new BufferedReader(new InputStreamReader(resourceAsStream));
            String temp;
            while ((temp = reader.readLine()) != null) {
                englishDicHashSet.add(temp.toLowerCase());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean isInDict(String str){
        if (str.length() == 1) {
            return false;
        }
        return englishDicHashSet.contains(str.toLowerCase());
    }
}
