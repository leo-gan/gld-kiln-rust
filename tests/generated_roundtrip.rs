//! Generated structs encode in JSON Schema property order.

#[path = "generated/Point.rs"]
mod point;
#[path = "generated/Message.rs"]
mod message;

use message::Message;
use point::Point;

#[test]
fn point_optional_label_round_trip() {
    let bare = Point {
        x: 3,
        y: -4,
        label: None,
    };
    let raw = bare.encode().expect("encode");
    assert_eq!(raw, [3, 0, 0, 0, 252, 255, 255, 255, 0]);
    assert_eq!(bare.encoded_size(), raw.len());
    let back = Point::decode(&raw).expect("decode");
    assert_eq!(back.x, 3);
    assert_eq!(back.y, -4);
    assert_eq!(back.label, None);

    let named = Point {
        x: 1,
        y: 2,
        label: Some("pq".into()),
    };
    let raw = named.encode().expect("encode");
    assert_eq!(named.encoded_size(), raw.len());
    let back = Point::decode(&raw).expect("decode");
    assert_eq!(back.label.as_deref(), Some("pq"));
    assert_eq!(back.x, 1);
    assert_eq!(back.y, 2);
}

#[test]
fn message_nested_round_trip() {
    let msg = Message {
        f_bool: true,
        f_int32: -3,
        f_int64: 99,
        f_float64: -0.0,
        f_string: "hi".into(),
        f_bool_2: false,
        f_int32_2: 0,
        f_string_2: String::new(),
        meta: message::Meta {
            region: "west".into(),
            flag: true,
        },
    };
    assert!(msg.f_float64.is_sign_negative());
    let meta_raw = msg.meta.encode().expect("meta encode");
    let meta_back = message::Meta::decode(&meta_raw).expect("meta decode");
    assert_eq!(meta_back.region, "west");
    assert!(meta_back.flag);
    let raw = msg.encode().expect("encode");
    assert_eq!(msg.encoded_size(), raw.len());
    let back = Message::decode(&raw).expect("decode");
    assert!(back.f_bool);
    assert_eq!(back.f_int32, -3);
    assert_eq!(back.f_int64, 99);
    assert_eq!(back.f_float64.to_bits(), (-0.0f64).to_bits());
    assert_eq!(back.f_string, "hi");
    assert!(!back.f_bool_2);
    assert_eq!(back.f_int32_2, 0);
    assert_eq!(back.f_string_2, "");
    assert_eq!(back.meta.region, "west");
    assert!(back.meta.flag);
}
