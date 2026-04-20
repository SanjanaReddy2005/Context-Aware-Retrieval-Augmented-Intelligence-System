"use client";
import { Button } from "@/components/ui/button";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Bot, Loader2 } from "lucide-react";
import axios from "axios";
import { useState } from "react";

const pdfSchema = z.object({
  file: z.instanceof(File),
});

interface UploadFormProps {
  setConnection: (data: string) => void;
}

const UploadForm = ({ setConnection }: UploadFormProps) => {
  const [loading, setLoading] = useState(false);
  const form = useForm<z.infer<typeof pdfSchema>>({
    resolver: zodResolver(pdfSchema),
  });

  async function onSubmit(values: z.infer<typeof pdfSchema>) {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", values.file);
    console.log(formData);
    console.log(values.file);
    try {
      const response = await axios.post(
        "http://localhost:8000/api/upload-file",
        formData
      );
      console.log(response.data.message);
      setConnection(response.data.message);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  }

  return (
    <div>
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
          <div className="">
            <FormField
              control={form.control}
              name="file"
              render={({ field: { value, onChange, ...fieldProps } }) => (
                <FormItem>
                  <FormLabel className="text-gray-700 dark:text-white">File</FormLabel>
                  <FormControl>
                    <Input
                      {...fieldProps}
                      type="file"
                      className="dark:bg-gray-700 dark:text-white"
                      accept=".pdf"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        onChange(file);
                      }}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
          <Button
            disabled={loading}
            type="submit"
            className="text-lg px-3 py-4 gap-2 bg-gradient-to-br dark:text-white from-amber-300 to to-orange-600 hover:scale-105 transition-all duration-300"
          >
            Upload
            {!loading && <Bot size={24} className="hover:animate-spin" />}
            {loading && <Loader2 size={24} className="animate-spin" />}
          </Button>
        </form>
      </Form>
    </div>
  );
};

export default UploadForm;
