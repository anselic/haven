//! Escape Haven code and add semantic spans for the static HTML renderer.
//! This scanner tolerates incomplete examples, unlike the compiler parser.

pub(super) fn haven(code: &str) -> String {
    let mut output = String::with_capacity(code.len());
    let mut offset = 0;
    while offset < code.len() {
        let remaining = &code[offset..];
        let bytes = remaining.as_bytes();
        let (len, class) = if remaining.starts_with("//") {
            (
                remaining.find('\n').unwrap_or(remaining.len()),
                Some("comment"),
            )
        } else if bytes[0] == b'"' {
            let mut end = 1;
            while end < bytes.len() {
                if bytes[end] == b'\\' {
                    end = (end + 2).min(bytes.len());
                } else if bytes[end] == b'"' {
                    end += 1;
                    break;
                } else {
                    end += 1;
                }
            }
            (end, Some("string"))
        } else if bytes[0].is_ascii_digit() {
            let len = bytes
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_' || **b == b'.')
                .count();
            (len, Some("number"))
        } else if bytes[0].is_ascii_alphabetic() || bytes[0] == b'_' {
            let len = bytes
                .iter()
                .take_while(|b| b.is_ascii_alphanumeric() || **b == b'_')
                .count();
            let word = &remaining[..len];
            let class = match word {
                "true" | "false" => Some("constant"),
                "let" | "if" | "else" | "return" | "while" | "for" | "break" | "continue"
                | "proc" | "extern" | "const" | "struct" | "enum" | "import" | "pub" | "match"
                | "extend" | "trait" | "where" | "type" => Some("keyword"),
                "void" | "bool" | "str" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32"
                | "u64" | "f32" | "f64" => Some("type"),
                _ if word.starts_with(|c: char| c.is_ascii_uppercase()) => Some("type"),
                _ => None,
            };
            (len, class)
        } else {
            (remaining.chars().next().unwrap().len_utf8(), None)
        };
        let fragment = &remaining[..len];
        if let Some(class) = class {
            output.push_str("<span class=\"syntax-");
            output.push_str(class);
            output.push_str("\">");
        }
        escape_into(&mut output, fragment);
        if class.is_some() {
            output.push_str("</span>");
        }
        offset += len;
    }
    output
}

fn escape_into(output: &mut String, value: &str) {
    for character in value.chars() {
        match character {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&#39;"),
            other => output.push(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::haven;

    #[test]
    fn highlights_tokens_and_escapes_source() {
        let html = haven("pub proc f(x: i32) bool { return x < 3; } // <unsafe>\n\"a&b\"");
        assert!(html.contains("<span class=\"syntax-keyword\">proc</span>"));
        assert!(html.contains("<span class=\"syntax-type\">i32</span>"));
        assert!(html.contains("<span class=\"syntax-number\">3</span>"));
        assert!(html.contains("<span class=\"syntax-comment\">// &lt;unsafe&gt;</span>"));
        assert!(html.contains("<span class=\"syntax-string\">&quot;a&amp;b&quot;</span>"));
        assert!(!html.contains("<unsafe>"));
    }
}
