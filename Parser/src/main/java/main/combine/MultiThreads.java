package main.combine;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;

public class MultiThreads {
    public static void main(String[] args) throws Exception {
        String path = "C:/Users/hu/Desktop/stat/acm/acm_javas/";
        String dstDirName = "C:/Users/hu/Desktop/stat/acm/res";
        parseDir(path, dstDirName);
    }

    static class Task implements Runnable {
        private final String path;
        private String result;

        public Task(String path) {
            this.path = path;
        }

        public String getResult() {
            return result;
        }

        @Override
        public void run() {
            this.result = new OneFileParsing(this.path).parse();
        }
    }

    public static void parseDir(String sourceDirName, String dstDirName) throws Exception {
        long startTime = System.currentTimeMillis();
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        File[] files = new File(sourceDirName).listFiles();

        assert files != null;
        for (File file : files) {
            String path = file.getAbsolutePath();
            executor.submit(() -> {
                Task task = new Task(path);
                Thread thread = new Thread(task);
                thread.start();
                try {
                    thread.join(3600000);
                    String fileName = new File(path).getName();
                    String result = task.getResult();
                    File targetFile;
                    if (result.startsWith("EXCEPTION")) {
                        targetFile = new File(dstDirName + "_pro/Res" + "_" + fileName.substring(0, fileName.lastIndexOf(".")));
                    } else {
                        targetFile = new File(dstDirName + "/Res" + "_" + fileName.substring(0, fileName.lastIndexOf(".")));
                    }
                    FileUtils.writeStringToFile(targetFile, result, "utf-8");
                } catch (InterruptedException | IOException e) {
                    System.err.println("EXCEPTION " + path + " " + e);
                } finally {
                    thread.stop();
                }
            });
        }

        executor.shutdown();
        ThreadPoolExecutor threadPoolExecutor = (ThreadPoolExecutor) executor;
        while (threadPoolExecutor.getCompletedTaskCount() < files.length) {
            Thread.sleep(10000);
            long duration = (System.currentTimeMillis() - startTime) / 1000;
            long parsed = threadPoolExecutor.getCompletedTaskCount();
            System.out.println("\r" + duration + "s\t" + parsed + "\t" + parsed / duration);
        }
    }

}


